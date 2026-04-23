from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .Void_node import (Void_SM_Model, Void_SM_VAE, Void_SM_Clip,
    Void_LATENTS, Void_SM_KSampler, Void_Encoder, Void_Vae_Decoder,Void_Selectpoints,
    Void_GetQuadMask)

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
