 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
from .model_loader_utils import (clear_comfyui_cache,save_lat_emb,read_lat_emb,load_T5_model,get_quadmask,perpare_masks_latents,encode_prompt,
    re_save_video,load_model,load_vae,decode_latents,prepare_latents,selectpoints
    )
from einops import rearrange
from .void_model.videox_fun.utils.fp8_optimization import convert_weight_dtype_wrapper
from .void_model.videox_fun.utils.utils import get_video_mask_input, save_videos_grid, save_inout_row,get_video_mask_input_
from diffusers import DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir
weigths_sam2_current_path = os.path.join(folder_paths.models_dir, "sam2")
if not os.path.exists(weigths_sam2_current_path):
    os.makedirs(weigths_sam2_current_path)
folder_paths.add_model_folder_path("sam2", weigths_sam2_current_path) #  sam2 dir

class Void_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Void_SM_Model",
            display_name="Void_SM_Model",
            category="Void_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("void_pass1",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("void_pass2",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls,dit,gguf,void_pass1,void_pass2) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        void_pass1_path=folder_paths.get_full_path("diffusion_models", void_pass1) if void_pass1 != "none" else None
        void_pass2_path=folder_paths.get_full_path("diffusion_models", void_pass2) if void_pass2 != "none" else None
        model= load_model(dit_path,gguf_path,void_pass1_path or void_pass2_path)
        return io.NodeOutput(model)

class Void_SM_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="Void_SM_VAE",
            display_name="Void_SM_VAE",
            category="Void_SM",
            inputs=[
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[io.Vae.Output(display_name="vae"),],
            )
    @classmethod
    def execute(cls,vae ) -> io.NodeOutput:
        clear_comfyui_cache()
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        vae=load_vae(vae_path,device) 
        return io.NodeOutput(vae)
    
class Void_SM_Clip(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="Void_SM_Clip",
            display_name="Void_SM_Clip",
            category="Void_SM",
            inputs=[
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("clip") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf") ),
            ],
            outputs=[io.Clip.Output(display_name="clip"),],
            )
    @classmethod
    def execute(cls,clip,gguf ) -> io.NodeOutput:
        clear_comfyui_cache()
        safetensors_path=folder_paths.get_full_path("clip", clip) if clip != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        clip=load_T5_model(safetensors_path)     
        return io.NodeOutput(clip)

class Void_Vae_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="Void_Vae_Decoder",
            display_name="Void_Vae_Decoder",
            category="Void_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latents",),  
                io.Boolean.Input("enable_tiling",default=True),
            ],
            outputs=
            [io.Image.Output(display_name="image"),],
            )
    @classmethod
    def execute(cls,vae,latents,enable_tiling ) -> io.NodeOutput:
        clear_comfyui_cache()
        image=decode_latents(vae,latents,enable_tiling)
        return io.NodeOutput(image)


        
class Void_LATENTS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Void_LATENTS",
            display_name="Void_LATENTS",
            category="Void_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("images"),
                io.Image.Input("quadmask_video"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("width", default=672, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=384, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("num_frames", default=197, min=8, max=1000,display_mode=io.NumberDisplay.number),
                io.Int.Input("temporal_window_size", default=85, min=8, max=1000,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("save_lat",default=True),
                
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                ],
            )
    @classmethod
    def execute(cls,vae,images,quadmask_video,seed,width,height,num_frames,temporal_window_size,save_lat) -> io.NodeOutput:
        clear_comfyui_cache() 
        # width=(width //32)*32 if width % 32 != 0  else width 
        # height=(height //32)*32 if height % 32 != 0  else height
       
        video_length = int((num_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
        input_video, input_video_mask,  =get_video_mask_input_(images,quadmask_video,(width,height),max_video_length=video_length,
            temporal_window_size=temporal_window_size,use_quadmask=True,)
        print(f"video_shape: {input_video.shape}") #video_shape: torch.Size([1, 3, 85, 672, 384])
        video_length = input_video.shape[2]
        
        image_processor = VaeImageProcessor(vae_scale_factor=8)
        init_video = image_processor.preprocess(rearrange(input_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
        init_video = init_video.to(dtype=torch.float32)
        init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)

        scheduler = DDIMScheduler.from_pretrained(os.path.join(node_cr_path,"CogVideoX-Fun-V1.5-5b-InP"), subfolder="scheduler")
        # MAX_VIDEO_LENGTH = 197
        # video_length = int((MAX_VIDEO_LENGTH - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
        generator = torch.Generator(device=device).manual_seed(seed)

        latents_outputs=prepare_latents( vae,scheduler,
            1,
            16,
            height,
            width,
            video_length, # need check
            torch.bfloat16,
            device,
            generator,
            latents=None,
            video=init_video,
            timestep=None,
            is_strength_max=True,
            return_noise=False,
            return_video_latents=False,)
        
        lat=latents_outputs[0]
        print(f"lat.shape: {lat.shape}") #lat.shape: torch.Size([1, 22, 16, 48, 84])
        
        inpaint_latents=perpare_masks_latents(vae,input_video_mask,input_video,init_video,lat,height,width,generator,device)

        print(f"inpaint_latents.shape: {inpaint_latents.shape}") #inpaint_latents.shape: torch.Size([1, 22, 32, 48, 84])
        latent={"samples":lat,"width":width,"height":height,"inpaint_latents":inpaint_latents,"num_frames":video_length}
        if save_lat:
            save_lat_emb("latent",latent)
        return io.NodeOutput(latent)
    
class Void_GetQuadMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Void_GetQuadMask",
            display_name="Void_GetQuadMask",
            category="Void_SM",
            inputs=[
                io.Combo.Input("sam2",options= ["none"] + folder_paths.get_filename_list("sam2")),
                io.Combo.Input("sam3",options= ["none"] + folder_paths.get_filename_list("sam2")),
                io.Float.Input("confidence_threshold",default=0.1,min=0,max=1.0,step=0.1,display_mode=io.NumberDisplay.number),
                io.String.Input("local_vllm",default="qwen3.5:9b",multiline=False),
                io.String.Input("config_path",default=".video_config.json",),
               
               
            ],
            outputs=[
                io.String.Output(display_name="quadmask_video"),
                ],
            )
    @classmethod
    def execute(cls,sam2,sam3,confidence_threshold,local_vllm,config_path,) -> io.NodeOutput:
        clear_comfyui_cache()
        sam2_path=folder_paths.get_full_path("sam2", sam2) if sam2 != "none" else None
        sam3_path=folder_paths.get_full_path("sam2", sam3) if sam3 != "none" else None
        sam2_config=os.path.join(node_cr_path,"sam2_hiera_l.yaml")
        quadmask_video=get_quadmask(sam2_path,config_path,sam2_config,model="gemini-3-pro-preview",
                                    segmentation_model="sam3",ckpt_path=sam3_path,confidence_threshold=confidence_threshold,vllm_model=local_vllm) 
        return io.NodeOutput(quadmask_video)


class Void_Selectpoints(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Void_Selectpoints",
            display_name="Void_Selectpoints",
            category="Void_SM",
            inputs=[
                io.Video.Input("video"),
                io.String.Input("prompt",multiline=True,default="A ball rolls off the table." ),
            ],
            outputs=[
                io.String.Output(display_name="config_path"),
                ],
            )
    @classmethod
    def execute(cls,video,prompt) -> io.NodeOutput:
        clear_comfyui_cache()
        video_path,video_dir=re_save_video(video, "auto", "quadmask", "auto")
        config_path=selectpoints(video_path,video_dir,prompt) 
        config_path=config_path.replace("video_config","video_config_points")
        return io.NodeOutput(config_path)


class Void_Encoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="Void_Encoder",
            display_name="Void_Encoder",
            category="Void_SM",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt",multiline=True,default="A ball rolls off the table." ),
                io.Boolean.Input("save_emb",default=True),
                io.Combo.Input("infer_device",options=['cuda','cpu',], ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                ],
            )
    @classmethod
    def execute(cls,clip,prompt,save_emb,infer_device) -> io.NodeOutput:
        clear_comfyui_cache()
        if infer_device=="cuda":
            clip.to(infer_device)
        emb,_=encode_prompt(clip,prompt,device=torch.device(infer_device),dtype=torch.bfloat16)
        if infer_device=="cuda":
            clip.to("cpu")
        #print(f"positive.shape: {emb.shape}") # torch.Size([1, 226, 4096])
        emb=[[emb.to(device),{}]]
        if save_emb:
            save_lat_emb("embeds",emb)
        return io.NodeOutput(emb)

class Void_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Void_SM_KSampler",
            display_name="Void_SM_KSampler",
            category="Void_SM",
            inputs=[
                io.Model.Input("model"),     
                io.Int.Input("steps", default=20, min=1, max=nodes.MAX_RESOLUTION,step=1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("offload", default=True),
                io.Int.Input("offload_block_num", default=1, min=1, max=48,step=1,display_mode=io.NumberDisplay.number),
                io.Latent.Input("latent",optional=True),  
                io.Conditioning.Input("positive",optional=True),

            ], 
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )
    @classmethod
    def execute(cls, model,steps,offload,offload_block_num,latent=None,positive=None) -> io.NodeOutput:
        convert_weight_dtype_wrapper(model.transformer, torch.bfloat16)
        if offload:
            from diffusers.hooks import apply_group_offloading
            apply_group_offloading(model.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=offload_block_num,)
        else:
            model.to(device)

        if positive is  None:
            positive=read_lat_emb("embeds",device)
        if latent is None:
            latent=read_lat_emb("latents",device)
        
        with torch.no_grad():
            sample = model(
                None,
                num_frames=latent["num_frames"],
                negative_prompt=None,
                height=latent["height"],
                width=latent["width"],
                prompt_embeds=positive[0][0],
                output_type = "latent",
                guidance_scale=1.0,
                num_inference_steps=steps,
                video=None,
                mask_video=latent,
                strength=1.0,
                use_trimask=True,
                use_vae_mask=True,
            )

        print(f"Output shape: {sample.shape}") # torch.Size([1, 22, 16, 48, 84])
        output={"samples":sample}
        return io.NodeOutput(output)


class Void_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Void_SM_Model,
            Void_SM_VAE,
            Void_SM_Clip,
            Void_LATENTS,
            Void_SM_KSampler,
            Void_Encoder,
            Void_Vae_Decoder,
            Void_GetQuadMask,
            Void_Selectpoints,
        ]
async def comfy_entrypoint() -> Void_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return Void_SM_Extension()
