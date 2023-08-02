# Get attention maps using Dino
"""
    Get saliency attention maps using Dino.
    
    DEPRECATED: Please do not use this script. It is not maintained.
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
from dino_extractor import ViTExtractor
from torch.nn import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import einops as ein
import wandb
import tyro
from dataclasses import dataclass
import time
import joblib
from tqdm.auto import tqdm
from typing import Union, Literal, Tuple, List
import traceback
# Library imports
from dvgl_benchmark.datasets_ws import BaseDataset
from configs import ProgArgs, BaseDatasetArgs
from configs import device


# %%
@dataclass
class LocalArgs:
    # Program arguments
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-Descs", 
            wandb_group="Attention-Maps")
    # Experiment ID
    exp_id: Union[bool,str] = False
    """
        Experiment ID. If False, then experiment results are not saved
        to disk. The folder where images are saved (if 'str') is
        `cache_dir/experiments/exp_id/images`. If 'True', then
        `cache_dir/images` is used.
    """
    # Dino parameters
    model_type: Literal["dino_vits8", "dino_vits16", "dino_vitb8", 
            "dino_vitb16", "vit_small_patch8_224", 
            "vit_small_patch16_224", "vit_base_patch8_224", 
            "vit_base_patch16_224"] = "dino_vits16"
    """
        Model for Dino to use as the base model.
    """
    down_scale_res: Tuple[int, int] = (224, 298)
    """
        Downscale resolution for DINO features (before extracting).
        Note that this is reduced to the nearest multiple of the patch
        size.
    """
    sub_sample: int = 10
    """
        Subsample the dataset by this factor for visualizing the
        attention maps.
    """
    sub_sample_attentions: int = 2
    """
        Subsample the attention maps by this factor. This is used when
        the stride is not equal to patch size (overlapping patches).
        Say patch size = 8 and stride = 4, then this should be 2
        because every second patch is non-overlapping (should be shown
        in the image). Basically, `Patch = (this_value * stride)`.
        If you want to visualize the entire attention map without any 
        aliasing (but possibly with a different size than the image), 
        then set this to 1.
    """
    show_qual: bool = False
    """
        If False, then no window is shown and the images are saved to 
        disk. If True, and `exp_id` is False, then images are shown
        using matplotlib. If True, and `exp_id` is a string, then
        images are saved to disk (no matplotlib window is shown).
    """


# %%

def main(largs: LocalArgs):
    # Wandb
    if largs.prog.use_wandb:
        wandb_run = wandb.init(project=largs.prog.wandb_proj,
                entity=largs.prog.wandb_entity, 
                group=largs.prog.wandb_group, config=largs)
    
    # Dataset
    bd_args = BaseDatasetArgs()
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}")
    # Dataset and extractor
    ds = BaseDataset(bd_args, ds_dir, ds_name, "test")
    
    # Shorthands
    ss_a = largs.sub_sample_attentions
    # Ensure that the save folder exists
    save_dir = largs.prog.cache_dir
    if type(largs.exp_id) == str:
        save_dir = f"{save_dir}/experiments/{largs.exp_id}"
    elif largs.exp_id == False or largs.exp_id is None:
        save_dir = None
        print("Experiment ID is False or None, not saving images")
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(f"{save_dir}/images", exist_ok=True)
            print(f"Created directory: {save_dir}")
        else:
            print(f"Directory already exists: {save_dir}!")
    # Main Dino extractor
    dino = ViTExtractor(largs.model_type, 4, device=device)
    for k in tqdm(range(0, len(ds), largs.sub_sample)):
        # Crop image to Dino patch size
        img = ds[k][0]
        img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
        img = F.interpolate(img, largs.down_scale_res)
        w = img.shape[3] - img.shape[3] % dino.p
        h = img.shape[2] - img.shape[2] % dino.p
        img_c = img[:, :, :h, :w]   # Cropped for Dino patch size
        w_f, h_f = img_c.shape[3] // dino.p, img_c.shape[2] // dino.p
        # Get attentions for all heads
        with torch.no_grad():
            res = dino._extract_features(img_c, [11], 'attn')
        nh = res[0].shape[1]    # Number of heads
        attentions = res[0][0, :, 0, 1:].reshape(nh, 
                                            *dino.num_patches)
        # Patch = ss_a * stride     Sub-sampled attention map
        attentions_ss = attentions[:, ::ss_a, ::ss_a]
        attentions_img_fs = F.interpolate(attentions_ss.unsqueeze(0),
                            scale_factor=dino.p, mode="nearest")[0]
        # Save attentions
        fig = plt.figure(figsize=(30, 3), dpi=300)
        plt_title = "Image: " \
            + ("database" if k < ds.database_num else "query") \
            + f" {os.path.basename(ds.images_paths[k])}"
        fig.suptitle(plt_title)
        gs = fig.add_gridspec(1, 1+nh)
        ax = fig.add_subplot(gs[0, 0])
        img_u8 = img_c[0].cpu().numpy()
        img_u8 = (img_u8 - img_u8.min()) / (img_u8.max()-img_u8.min())
        img_u8 = (img_u8 * 255).astype(np.uint8).transpose(1, 2, 0)
        ax.imshow(img_u8)
        ax.set_axis_off()
        for i in range(nh):
            ax = fig.add_subplot(gs[0, i+1])
            ax.set_title(f"Head {i}")
            _im = ax.imshow(attentions_img_fs[i].cpu().numpy())
            ax.set_axis_off()
            fig.colorbar(_im)
        fig.set_tight_layout(True)
        if save_dir is not None:
            save_path = f"{save_dir}/images/{k}.png"
            fig.savefig(save_path)
            if largs.prog.wandb_save_qual:
                wandb.log({"Qual-Results": wandb.Image(save_path)})
        elif largs.show_qual:
            plt.show()
        plt.close(fig)
    
    results = {
        "DB-Name": str(ds_name),
        "Model-Type": str(largs.model_type),
        "Patch-Size": str(dino.p),
        "Num-Patches": str(dino.num_patches),
        "Stride": str(dino.stride),
        "Save-Dir": str(save_dir),
    }
    
    if save_dir is not None:
        save_res_file = f"{save_dir}/results.gz"
        print(f"Saved all images to {save_dir}/images")
        results["Imgs-Dir"] = f"{save_dir}/images"
        results["Args"] = str(largs)
        print(f"Saved all results to {save_res_file}")
        joblib.dump(results, save_res_file)
    
    # End
    if largs.prog.use_wandb:
        wandb.log(results)
        print(f"Finished WandB run: {wandb_run.name}")
        wandb.finish()


if __name__ == "__main__" and (not "ipykernel" in sys.argv[0]):
    # Parse arguments
    largs = tyro.cli(LocalArgs)
    print(f"Arguments: {largs}")
    _start = time.time()
    try:
        main(largs)
    except:
        print("Unhandled exception")
        traceback.print_exc()
    finally:
        print(f"Total time: {time.time() - _start:.2f} seconds")
        exit(0)


# %%
# Experiments

# %%
# Replicating main from above

# %%
largs = LocalArgs(ProgArgs(vg_dataset_name="17places"), model_type="dino_vits16", sub_sample_attentions=4)

# %%
bd_args = BaseDatasetArgs()
ds_dir = largs.prog.data_vg_dir
ds_name = largs.prog.vg_dataset_name
print(f"Dataset directory: {ds_dir}")
print(f"Dataset name: {ds_name}")
# Dataset and extractor
ds = BaseDataset(bd_args, ds_dir, ds_name, "test")

ss_a = largs.sub_sample_attentions

# Main Dino extractor
dino = ViTExtractor(largs.model_type, 4, device=device)

# %%
k = 10

img = ds[k][0]
img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
img = F.interpolate(img, largs.down_scale_res)
w = img.shape[3] - img.shape[3] % dino.p
h = img.shape[2] - img.shape[2] % dino.p
img_c = img[:, :, :h, :w]   # Cropped for Dino patch size
print(f"Image cropped from {img.shape} to {img_c.shape}")
w_f, h_f = img_c.shape[3] // dino.p, img_c.shape[2] // dino.p
# Get attentions for all heads
with torch.no_grad():
    res = dino._extract_features(img_c, [11], 'attn')
nh = res[0].shape[1]    # Number of heads
attentions = res[0][0, :, 0, 1:].reshape(nh, 
                                    *dino.num_patches)
# Patch = ss_a * stride     Sub-sampled attention map
attentions_ss = attentions[:, ::ss_a, ::ss_a]
attentions_img_fs = F.interpolate(attentions_ss.unsqueeze(0),
                            scale_factor=dino.p, mode="nearest")[0]
print(f"Attention maps are of shape: {attentions_img_fs.shape}")

# %%
fig = plt.figure(figsize=(30, 3), dpi=300)
plt_title = "Image: " \
    + ("database" if k < ds.database_num else "query") \
    + f" {os.path.basename(ds.images_paths[k])}"
fig.suptitle(plt_title)
gs = fig.add_gridspec(1, 1+nh)
ax = fig.add_subplot(gs[0, 0])
img_u8 = img_c[0].cpu().numpy()
img_u8 = (img_u8 - img_u8.min()) / (img_u8.max()-img_u8.min())
img_u8 = (img_u8 * 255).astype(np.uint8).transpose(1, 2, 0)
ax.imshow(img_u8)
ax.set_axis_off()
for i in range(nh):
    ax = fig.add_subplot(gs[0, i+1])
    ax.set_title(f"Head {i}")
    _im = ax.imshow(attentions_img_fs[i].cpu().numpy())
    ax.set_axis_off()
    fig.colorbar(_im)
fig.set_tight_layout(True)
plt.show()

# %%
# TODO: Trying to integrate 'dino_repo_main' (original Dino attention)

# %%
bd_args = BaseDatasetArgs()
ds_dir = largs.prog.data_vg_dir
ds_name = largs.prog.vg_dataset_name
print(f"Dataset directory: {ds_dir}")
print(f"Dataset name: {ds_name}")
# Dataset and extractor
ds = BaseDataset(bd_args, ds_dir, ds_name, "test")


# %%
i = 10

img = ds[i][0]
img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
img = F.interpolate(img, largs.down_scale_res)

# %%


# %%
from dino_repo_main import vision_transformer as vits

# %%
patch_size = 16
v = vits.vit_small(patch_size).to(device)

# %%
w = img.shape[3] - img.shape[3] % patch_size
h = img.shape[2] - img.shape[2] % patch_size
img_cropped = img[:, :, :h, :w]
print(f"Image shape: {img_cropped.shape}")

# %%
img_u8 = img_cropped[0].cpu().numpy()
img_u8 = (img_u8 - img_u8.min()) / (img_u8.max() - img_u8.min())
img_u8 = (img_u8 * 255).astype(np.uint8).transpose(1, 2, 0)
plt.imshow(img_u8)

# %%
with torch.no_grad():
    res = v.get_last_selfattention(img_cropped)

print(f"Result shape: {res.shape}")
nh = res.shape[1]

w_f, h_f = img_cropped.shape[3] // patch_size, \
        img_cropped.shape[2] // patch_size

attentions = res[0, :, 0, 1:]
attentions_img = ein.rearrange(attentions, 
        "nh (h_f w_f) -> nh h_f w_f", h_f=h_f, w_f=w_f)
attentions_img_fs = F.interpolate(attentions_img[None, ...], 
        scale_factor=patch_size, mode="nearest")[0]


# %%
plt.imshow(attentions_img_fs[5].cpu().numpy())
plt.colorbar()
plt.show()

# %%
