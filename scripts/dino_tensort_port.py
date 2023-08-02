# Port Dino and Dino-v2 feature extraction network to TensorRT
"""
    
    
    TensorRT: https://github.com/NVIDIA/TensorRT
"""

# %%
import os
import sys
from pathlib import Path
# Set the './../' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')


# %%
import torch
from torch import nn
import torch_tensorrt
from torch.nn import functional as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
import tyro
from dataclasses import dataclass, field
from utilities import VLAD, get_top_k_recall, seed_everything
import einops as ein
import wandb
import matplotlib.pyplot as plt
import time
import joblib
import traceback
from tqdm.auto import tqdm
from dvgl_benchmark.datasets_ws import BaseDataset
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from typing import Union, Literal, Tuple, List
from utilities import DinoV2ExtractFeatures


# %%
# Experimental section

# %%

MT = Literal["dinov2_vits14", "dinov2_vitb14", 
        "dinov2_vitl14", "dinov2_vitg14"]   # Model types
FT = Literal["query", "key", "value", "token"]  # Facet types

# V2 wrapper
class DinoV2Wrapper(nn.Module):
    def __init__(self, model_type: MT, layer_num: int, 
                desc_facet: FT) -> None:
        """
            - model_type:   Dino model
            - layer_num:    Layer Number
            - desc_facet:   Descriptor facet from the layer
        """
        super().__init__()
        self.model_type: str = model_type
        self.dino_model: nn.Module = torch.hub.load(
                'facebookresearch/dinov2', self.model_type)
        self.facet: str = desc_facet
        self.layer: int = layer_num
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
            - img: Input image of shape [1, 3, H, W] (where H and W
                are suggested to be in multiples of 14)
        """
        res = self.dino_model(img)
        res = self._hook_out[:, 1:, ...]
        if self.facet in ["query", "key", "value"]:
            d_len = res.shape[2] // 3
            if self.facet == "query":
                res = res[:, :, :d_len]
            elif self.facet == "key":
                res = res[:, :, d_len:2*d_len]
            else:
                res = res[:, :, 2*d_len:]
        return res


# %%
img = torch.randn(1, 3, 224, 224)

# %%
md = DinoV2Wrapper("dinov2_vits14", 1, "query")

# %%
# Doesn't work
"""
    NotSupportedError: Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults:
"""
inputs = [
    torch_tensorrt.Input((1, 3, 244, 244), dtype=torch.float32)
]
trt_ts_module = torch_tensorrt.compile(md, inputs=inputs)


# %%
dino_model = torch.hub.load('facebookresearch/dinov2', 
        "dinov2_vits14")

# %%
_res = None
def _forward_hook(module, inputs, output):
    global _res
    _res = output

# %%
fh_handle = dino_model.blocks[1].attn.qkv\
        .register_forward_hook(_forward_hook)

# %%
res = dino_model(img)

# %%
input_data = torch.empty([1, 3, 224, 224])

# %%
trt_ts_module = torch.jit.script(dino_model)

# %%

