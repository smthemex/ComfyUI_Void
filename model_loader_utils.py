# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import comfy.model_management as mm
from PIL import Image
import numpy as np
from comfy.utils import common_upscale
import folder_paths
import time
from safetensors.torch import load_file
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
from diffusers import DDIMScheduler
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F
import json
import argparse
from .void_model.vlm_mask.stage1_sam2_segmentation import process_config as process_config_stage1
from .void_model.vlm_mask.stage2_vlm_analysis import process_config as process_config_stage2
from .void_model.vlm_mask.stage3a_generate_grey_masks import main as process_config_stage3
from .void_model.vlm_mask.stage3a_generate_grey_masks_v2 import main as process_config_stage3_v2
from .void_model.vlm_mask.stage4_combine_masks   import process_config as process_config_stage4

from transformers import T5Config
from .void_model.videox_fun.pipeline import CogVideoXFunInpaintPipeline
from .void_model.videox_fun.pipeline.pipeline_cogvideox_fun_inpaint import resize_mask,add_noise_to_reference_video
from .void_model.videox_fun.models import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    T5EncoderModel,
    T5Tokenizer,
)
cur_path = os.path.dirname(os.path.abspath(__file__))


def  re_save_video(video,codec,filename_prefix,format):
    from comfy_api.latest import  Types

    width, height = video.get_dimensions()
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )

    saved_metadata = None

    file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
    video_path=os.path.join(full_output_folder, file)
    filename = os.path.basename(video_path)
    folder_name, _ = os.path.splitext(filename)
    video_dir=os.path.join(full_output_folder,folder_name)
    os.makedirs(video_dir, exist_ok=True)
    video.save_to(
        os.path.join(video_dir, file),
        format=Types.VideoContainer(format),
        codec=codec,
        metadata=saved_metadata
    )
        # filepath_list.append(os.path.join(video_dir, file))
        # file_folder_list.append(video_dir)
    return os.path.join(video_dir, file),video_dir


def load_model(model_path,gguf_path,pt_path,dtype=torch.bfloat16):
    # normal dit
    # ctx = init_empty_weights if is_accelerate_available() else nullcontext
    # dit_config=CogVideoXTransformer3DModel.load_config(os.path.join(cur_path,"CogVideoX-Fun-V1.5-5b-InP/transformer"),low_cpu_mem_usage=False, use_vae_mask=True,)
    # with ctx():
    #     transformer=CogVideoXTransformer3DModel.from_config(dit_config, torch_dtype=dtype)
    # if model_path is not None:
    #     dit_state_dict = load_file(model_path)
    #     match_state_dict(transformer, dit_state_dict,show_num=10)
    #     transformer.load_state_dict(dit_state_dict, strict=False, assign=True)
    # elif gguf_path is not None:
    #     dit_state_dict=load_gguf_checkpoint(gguf_path)
    #     set_gguf2meta_model(transformer,dit_state_dict,dtype,torch.device("cpu"))
    # del dit_state_dict
    transformer=CogVideoXTransformer3DModel.from_pretrained(model_path, torch_dtype=dtype,low_cpu_mem_usage=True, use_vae_mask=True,
                config_file=os.path.join(cur_path,"CogVideoX-Fun-V1.5-5b-InP/transformer/config.json"))
    transformer.eval()
    param_name = "patch_embed.proj.weight"
    print(f"Loading VOID checkpoint from {pt_path}...")
    state_dict = load_file(pt_path)
    if state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1):
        latent_ch, feat_scale = 16, 8
        feat_dim = latent_ch * feat_scale
        new_weight = transformer.state_dict()[param_name].clone()
        new_weight[:, :feat_dim] = state_dict[param_name][:, :feat_dim]
        new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
        state_dict[param_name] = new_weight
        del new_weight
        print(f"Adapted {param_name} channels for VAE mask")
    m, u = transformer.load_state_dict(state_dict, strict=False)
    del state_dict
    print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

    tokenizer = T5Tokenizer.from_pretrained(os.path.join(cur_path,"CogVideoX-Fun-V1.5-5b-InP"), subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(os.path.join(cur_path,"CogVideoX-Fun-V1.5-5b-InP"), subfolder="scheduler")
    pipe = CogVideoXFunInpaintPipeline(
            tokenizer=tokenizer,
            text_encoder=None,
            vae=None,
            transformer=transformer,
            scheduler=scheduler,
        )
    return pipe


def load_T5_model(model_path):
    config=T5Config.from_pretrained(os.path.join(cur_path,"CogVideoX-Fun-V1.5-5b-InP"), subfolder="text_encoder",torch_dtype=torch.bfloat16)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        t5_model=T5EncoderModel(config)
    sd = load_file(model_path)
    if "encoder.embed_tokens.weight" not in sd:
        sd["encoder.embed_tokens.weight"]=sd["shared.weight"]
    match_state_dict(t5_model, sd,show_num=10)    
    t5_model.load_state_dict(sd, strict=False, assign=True) 
    t5_model.eval().to(torch.bfloat16)
    del sd
    return t5_model

def load_vae(vae_path,device):
    # Load vae
    vae_config=AutoencoderKLCogVideoX.load_config(os.path.join(cur_path,"CogVideoX-Fun-V1.5-5b-InP/vae"))
    vae=AutoencoderKLCogVideoX.from_config(vae_config,torch_dtype=torch.bfloat16)
    vae_state_dict = load_file(vae_path)

    # check state dict keys
    match_state_dict(vae, vae_state_dict,show_num=10)

    vae.load_state_dict(vae_state_dict, strict=False)
    vae.eval().to(device)
    del vae_state_dict
    #vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae").to(torch.bfloat16)
    return vae

def decode_latents(vae,latents: torch.Tensor,enable_tiling) -> torch.Tensor:
    latents=latents["samples"] if isinstance(latents, dict) else latents
    latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    latents = 1 / vae.config.scaling_factor * latents
    if enable_tiling:
        vae.enable_tiling()
    else:
        vae.disable_tiling()
    frames = vae.decode(latents.to(vae.dtype)).sample
    frames = (frames / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    frames = frames.cpu().float() 
    print(frames.shape)#(1, 3, 85, 384, 672)
    frames = rearrange(frames, "b c t h w -> (b t) h w c")
    return frames

def selectpoints(video_path,output_dir,prompt):
    data={
        "videos": [
            {
            "video_path": video_path,
            "output_dir": output_dir,
            "instruction": prompt,
            }
        ]
        }
    file_path = os.path.join(output_dir,"video_config.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved video config to {file_path}. Launching point selector GUI...")
    # from .void_model.vlm_mask.point_selector_gui import main_ 
    # main_(file_path)

    return file_path

def get_quadmask(sam2_path,config_path,sam2_config,model,segmentation_model,ckpt_path,confidence_threshold,vllm_model,device="cuda"):
    # config=os.path.join(cur_path,"void_model/vlm_mask/config.yaml")

    process_config_stage1(config_path,sam2_path,sam2_config,device) #black_mask.mp4

    process_config_stage2(config_path, model,vllm_model) #vlm_analysis.json

    args = argparse.Namespace(
        config=config_path,
        segmentation_model=segmentation_model,
        bpe_path=os.path.join(cur_path,"void_model/sam3/bpe_simple_vocab_16e6.txt.gz"),
        ckpt_path=ckpt_path,
        confidence_threshold=confidence_threshold,
    )
   
    process_config_stage3_v2(args) #grey_mask.mp4

    combined_frames=process_config_stage4(config_path) #quadmask_0.mp4

    return combined_frames
    

def prepare_mask_latents(vae,mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        add_noise_in_inpaint_model=True
        if mask is not None:
            mask = mask.to(device=device, dtype=vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim = 0)
            mask = mask * vae.config.scaling_factor

        if masked_image is not None:
            if add_noise_in_inpaint_model:
                masked_image = add_noise_to_reference_video(masked_image, ratio=noise_aug_strength)
            masked_image = masked_image.to(device=device, dtype=vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
            masked_image_latents = masked_image_latents * vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents

def perpare_masks_latents(vae,mask_video,video,init_video,latents,height,width,generator,device,use_trimask=True,use_vae_mask=True,binarize_mask=False,noise_aug_strength = 0.0563,
                          zero_out_mask_region=False,stack_mask=False, do_classifier_free_guidance=False ):
    num_channels_latents=16
    num_channels_transformer=33
    mask_processor = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False, do_binarize=False, do_convert_grayscale=True
        )
    masked_video_latents = None
    if (mask_video == 255).all():
        print("No masked region detected, skipping inpainting and using original latents.")
        mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
        masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

        mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
        masked_video_latents_input = (
            torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
        )
        inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
    else:
        # Prepare mask latent variables
        video_length = video.shape[2]
        mask_condition = mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
        if use_trimask:
            mask_condition = torch.where(mask_condition > 0.75, 1., mask_condition)
            mask_condition = torch.where((mask_condition <= 0.75) * (mask_condition >= 0.25), 127. / 255., mask_condition)
            mask_condition = torch.where(mask_condition < 0.25, 0., mask_condition)
        else:
            mask_condition = torch.where(mask_condition > 0.5, 1., 0.)
            
        mask_condition = mask_condition.to(dtype=torch.float32)
        mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

        if num_channels_transformer != num_channels_latents:
            mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
            if masked_video_latents is None:
                if zero_out_mask_region:
                    masked_video = init_video * (mask_condition_tile < 0.75) + torch.ones_like(init_video) * (mask_condition_tile > 0.75) * -1
                else:
                    masked_video = init_video
            else:
                masked_video = masked_video_latents

            mask_encoded, masked_video_latents = prepare_mask_latents(vae,
                1 - mask_condition_tile if use_vae_mask else None,
                masked_video,
                1,
                height,
                width,
                torch.bfloat16,
                device,
                generator,
                do_classifier_free_guidance,
                noise_aug_strength=noise_aug_strength,
            )
            if not use_vae_mask and not stack_mask:
                mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                if binarize_mask:
                    if use_trimask:
                        mask_latents = torch.where(mask_latents > 0.75, 1., mask_latents)
                        mask_latents = torch.where((mask_latents <= 0.75) * (mask_latents >= 0.25), 0.5, mask_latents)
                        mask_latents = torch.where(mask_latents < 0.25, 0., mask_latents)
                    else:
                        mask_latents = torch.where(mask_latents < 0.9, 0., 1.).to(mask_latents.dtype)
                scaling_factor=0.7
                mask_latents = mask_latents.to(masked_video_latents.device) * scaling_factor

                mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                mask = rearrange(mask, "b c f h w -> b f c h w")
            elif stack_mask:
                mask_latents = torch.cat([
                    torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
                    mask_condition[:, :, 1:],
                ], dim=2)
                mask_latents = mask_latents.view(
                    mask_latents.shape[0],
                    mask_latents.shape[2] // 4,
                    4,
                    mask_latents.shape[3],
                    mask_latents.shape[4],
                )
                mask_latents = mask_latents.transpose(1, 2)
                mask_latents = resize_mask(1 - mask_latents, masked_video_latents).to(latents.device, latents.dtype)
                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
            else:
                mask_input = (
                    torch.cat([mask_encoded] * 2) if do_classifier_free_guidance else mask_encoded
                )

            masked_video_latents_input = (
                torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
            )

            mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
            masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")

            # concat(binary mask, encode(mask * video))
            inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
        else:
            mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
            mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
            mask = rearrange(mask, "b c f h w -> b f c h w")

            inpaint_latents = None
    return inpaint_latents


def prepare_latents(vae,scheduler,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        video=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_video_latents=False,
    ):
        vae_scale_factor_temporal=4
        vae_scale_factor_spatial=8
        shape = (
            batch_size,
            (video_length - 1) // vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // vae_scale_factor_spatial,
            width // vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if return_video_latents or (latents is None and not is_strength_max):
            video = video.to(device=device, dtype=vae.dtype)
            
            bs = 1
            new_video = []
            for i in range(0, video.shape[0], bs):
                video_bs = video[i : i + bs]
                video_bs = vae.encode(video_bs)[0]
                video_bs = video_bs.sample()
                new_video.append(video_bs)
            video = torch.cat(new_video, dim = 0)
            video = video * vae.config.scaling_factor

            video_latents = video.repeat(batch_size // video.shape[0], 1, 1, 1, 1)
            video_latents = video_latents.to(device=device, dtype=dtype)
            video_latents = rearrange(video_latents, "b c f h w -> b f c h w")

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else scheduler.add_noise(video_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * scheduler.init_noise_sigma

        # scale the initial noise by the standard deviation required by the scheduler
        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_video_latents:
            outputs += (video_latents,)

        return outputs

def get_t5_prompt_embeds(text_encoder,tokenizer,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):


        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

def encode_prompt(text_encoder,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = False,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        tokenizer = T5Tokenizer.from_pretrained(os.path.join(cur_path,"CogVideoX-Fun-V1.5-5b-InP"), subfolder="tokenizer")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = get_t5_prompt_embeds(text_encoder,tokenizer,
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds


def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list

def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "bilinear", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "bilinear", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img

def read_lat_emb(prefix, device):
    if prefix =="embeds":
        if not  os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_embeds_Viod_sm.pt")):
            raise Exception("No backup prompt embeddings found. Please run Viod_SM_ENCODER node first.")
        else:
            prompt_embeds=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_embeds_Viod_sm.pt"),weights_only=False)

        positive=[[prompt_embeds[0][0].to(device,torch.bfloat16),{}]]
        return positive
    
    elif prefix =="latents":
        if not  os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_latents_Viod_sm.pt")) :
            raise Exception("No backup latents found. Please run Viod_SM_KSampler node first.")
        else:
            video_latents=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_latents_Viod_sm.pt"),weights_only=False)
        video_latents["samples"]=video_latents["samples"].to(device,torch.bfloat16)
        video_latents["inpaint_latents"]=video_latents["inpaint_latents"].to(device)
        print(f"video shape: {video_latents['samples'].shape}")


        return video_latents
    
def  save_lat_emb(save_prefix,data1):
    data1_prefix="raw_embeds_Viod" if save_prefix == "embeds" else "raw_latents_Viod"
    default_data1_path = os.path.join(folder_paths.get_output_directory(),f"{data1_prefix}_sm.pt")
    prefix = str(int(time.time()))
    if os.path.exists(default_data1_path): # use a different path if the file already exists
        default_data1_path=os.path.join(folder_paths.get_output_directory(),f"{data1_prefix}_sm_{prefix}.pt")
    torch.save(data1,default_data1_path)
   

def map_0_1_to_neg1_1(t):
    """
    接受 torch.Tensor 或可转 torch.Tensor 的输入。
    可处理形状: H,W,C 或 B,H,W,C 或 B,T,H,W,C（会按元素处理）。
    确保 float，0..255 -> 0..1，再把 0..1 -> -1..1（如果已经在 -1..1 则不变）。
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()
    # 处理 0..255 的情况
    try:
        vmax = float(t.max())
    except Exception:
        vmax = 1.0
    if vmax > 2.0:
        t = t / 255.0
    # 若当前处于 0..1 范围，则映射到 -1..1
    try:
        vmin = float(t.min())
        vmax = float(t.max())
    except Exception:
        vmin, vmax = -1.0, 1.0
    if vmin >= 0.0 and vmax <= 1.1:
        t = t * 2.0 - 1.0
    return t

def map_neg1_1_to_0_1(t):
    """
    接受 torch.Tensor 或可转 torch.Tensor 的输入。
    可处理形状: H,W,C 或 B,H,W,C 或 B,T,H,W,C（会按元素处理）。
    返回 float tensor，范围 0..1。
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()
    # map -1..1 -> 0..1
    t = (t + 1.0) * 0.5
    # 限幅到 [0,1]
    t = t.clamp(0.0, 1.0)
    # 保持在 cpu 端，调用方可决定是否转 device/dtype
    return t

def load_gguf_checkpoint(gguf_checkpoint_path):

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
  
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        quant_type = tensor.tensor_type

        
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data) #tensor.data.copy()
 
        parsed_parameters[name.replace("model.", "")] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
        del tensor,weights
        if i > 0 and i % 1000 == 0:  # 每1000个tensor执行一次gc
            logger.info(f"Processed {i}tensors...")
            gc.collect()
    del reader
    gc.collect()
    return parsed_parameters

def set_gguf2meta_model(meta_model,model_state_dict,dtype,device):
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
    hf_quantizer = GGUFQuantizer(quantization_config=g_config)
    hf_quantizer.pre_quantized = True


    hf_quantizer._process_model_before_weight_loading(
        meta_model,
        device_map={"": device} if device else None,
        state_dict=model_state_dict
    )
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    x,y=load_model_dict_into_meta(
        meta_model, 
        model_state_dict, 
        hf_quantizer=hf_quantizer,
        device_map={"": device} if device else None,
        dtype=dtype
    )
    print(x,"offload_index")
    print(y,"state_dict_index")

    hf_quantizer._process_model_after_weight_loading(meta_model)

    
    del model_state_dict
    gc.collect()
    return meta_model.to(dtype=dtype)

def match_state_dict(meta_model, sd,show_num=10):

    meta_model_keys = set(meta_model.state_dict().keys())   
    state_dict_keys = set(sd.keys())

    # 打印匹配的键的数量
    matching_keys = meta_model_keys.intersection(state_dict_keys)
    print(f"Matching keys count: {len(matching_keys)}")
    
    # 打印不在 meta_model 中但在 state_dict 中的键（多余键）
    extra_keys = state_dict_keys - meta_model_keys
    if extra_keys:
        print(f"Extra keys in state_dict (not in meta_model): {len(extra_keys)}")
        for key in list(extra_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 打印不在 state_dict 中但在 meta_model 中的键（缺失键）
    missing_keys = meta_model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state_dict (not in state_dict): {len(missing_keys)}")
        for key in list(missing_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 如果需要，也可以打印部分匹配的键
    print(f"Sample matching keys: {list(matching_keys)[:5]}")