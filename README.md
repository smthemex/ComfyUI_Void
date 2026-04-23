# ComfyUI_Void
[Void](https://github.com/Netflix/void-model) :Video Object and Interaction Deletion


---
## 1. Installation  

In the `./ComfyUI/custom_nodes` directory, run:

```
git clone https://github.com/smthemex/ComfyUI_Void

```

## 2. Requirements  

```
pip install -r requirements.txt
```
need llama local api  qwen3.5-9b to run if not has gemma api # 如果没有gemma api 需要安装本地llama 并调用本地  qwen3.5-9b llama模型 

## 3. Checkpoints 

* 3.1 [pass1 or pass2](https://huggingface.co/netflix/void-model/tree/main) 
* 3.2 [sam2_hiera_large.pt](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation)
* 3.3 [sam3](https://huggingface.co/facebook/sam3/tree/main)
* 3.4 [transformer and vae and text encoder](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP)
* 3.5 llama qwen3.5 9B

```
├── ComfyUI/models/diffusion_models
|     ├── CogVideoX-Fun-V1.5-5b-InP-Transformer.safetensors
|     ├── void_pass1.safetensors
├── ComfyUI/models/vae
|     ├──CogVideoX-Fun-V1.5-5b-InP-VAE.safetensors
├── ComfyUI/models/sam2 
|     ├──SAM3.safetensors
|     ├──sam2_hiera_large.pt
├── ComfyUI/models/clip
|     ├──CogVideoX-Fun-V1.5-5b-InP-TextEncoder.safetensors
├── llama
|     ├──qwen3.5-9b.gguf 
```

## 4. Usage

* read the workflow note  仔细阅读工作流的note说明


Example
----

![](https://github.com/smthemex/ComfyUI_Void/blob/main/example_workflows/example.png)


# Citation
```
@misc{motamed2026void,
  title={VOID: Video Object and Interaction Deletion},
  author={Saman Motamed and William Harvey and Benjamin Klein and Luc Van Gool and Zhuoning Yuan and Ta-Ying Cheng},
  year={2026},
  eprint={2604.02296},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.02296}
}
```
