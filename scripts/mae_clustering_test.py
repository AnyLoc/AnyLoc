# Given an image, view and cluster the MAE descriptors
"""
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
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as T
from PIL import Image
import seaborn as sns
import models_mae
import einops as ein
import cv2 as cv
from fast_pytorch_kmeans import KMeans
import matplotlib.pyplot as plt
from typing import Literal
from configs import base_dataset_args as bd_args
from dvgl_benchmark.datasets_ws import BaseDataset
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens


# %%
data_vg_dir: Path = f"{lib_path}/datasets_vg/datasets"
vg_dataset_name: Literal["st_lucia", "pitts30k", "17places", 
        "nordland", "tokyo247", "baidu_datasets", "Oxford", 
        "gardens"] = "Oxford"
data_split: Literal["train", "test", "val"] = "test"
mae_model: Literal["mae_vit_base_patch16", 
        "mae_vit_large_patch16", "mae_vit_huge_patch14"] = \
                "mae_vit_large_patch16"
ckpt_path: Path = f"{lib_path}/models/mae/"\
        "mae_visualize_vit_large_ganloss.pth"
sample_i: int = 10


# %%
_ex = lambda x: os.path.realpath(os.path.expanduser(x))
ds_dir, ds_name = data_vg_dir, vg_dataset_name
print(f"Dataset directory: {ds_dir}")
print(f"Dataset name: {ds_name}, split: {data_split}")
# Load dataset
if ds_name=="baidu_datasets":
    vpr_ds = Baidu_Dataset(bd_args, ds_dir, ds_name, 
                        data_split)
elif ds_name=="Oxford":
    vpr_ds = Oxford(ds_dir)
elif ds_name=="gardens":
    vpr_ds = Gardens(bd_args,ds_dir,ds_name,data_split)
else:
    vpr_ds = BaseDataset(bd_args, ds_dir, ds_name, 
                    data_split)
ckpt_path = _ex(ckpt_path)


# %%
img_fname = vpr_ds.get_image_paths()[sample_i]
print(f"Image file name: {img_fname}")
img_pil = Image.open(img_fname)
img_np = np.array(img_pil)
img_pt = vpr_ds[sample_i][0]

# %%
# img_pt_ = T.center_crop(img_pt[None, ...], min(img_pt.shape[1:]))
# img_pt_ = F.interpolate(img_pt_, (224, 224))
img_pt_ = T.center_crop(img_pt[None, ...], 224)

# %%
model: models_mae.MaskedAutoencoderViT = getattr(models_mae, 
        mae_model)(ret_latents=True)
ckpt = torch.load(ckpt_path)
msg = model.load_state_dict(ckpt["model"])
print(f"Model loaded: {msg}")
model = model.to("cuda")

# %%
torch.set_grad_enabled(False)

# %%
_, _, _, latents = model(img_pt_.to("cuda"), mask_ratio=0.0)

# %%
c, h, w = img_pt.shape
h_new, w_new = map(lambda x: int((x//224)*224), [h, w])
img_pt_ = T.resize(img_pt[None, ...], (h_new, w_new))
print(f"Resized: [{h}, {w}] -> [{h_new}, {w_new}]")

# %%
img_patches = ein.rearrange(img_pt_, 
        "b c (nh h) (nw w) -> (nh nw) b c h w", b=1,
        nh=h_new//224, nw=w_new//224)

# %%
latents_patches = []
for sub_img in img_patches:
    _, _, _, latents = model(sub_img.to("cuda"), mask_ratio=0.0)
    latents_patches.append(latents[:, 1:, :].detach().cpu())
latents_patches = torch.cat(latents_patches, dim=0)
# latents_patches = ein.rearrange(latents_patches, "p n d -> (p n) d")

# %%
latents_patches_img = ein.rearrange(latents_patches, 
        "(nh nw) (ph pw) d -> 1 d (nh ph) (nw pw)", 
        nh=h_new//224, nw=w_new//224, ph=224//16, pw=224//16)

# %%
lpimg_up = F.interpolate(latents_patches_img, (448, 896), 
        mode="nearest")
lpimg_up = F.normalize(lpimg_up, dim=1)

# %%
lpimg_up_descs = ein.rearrange(lpimg_up, "1 d h w -> (h w) d")

# %%
kmeans = KMeans(n_clusters=8)
labels = kmeans.fit_predict(lpimg_up_descs)

# %%
img_labels = ein.rearrange(labels, "(h w) -> 1 1 h w", 
        h=lpimg_up.shape[2], w=lpimg_up.shape[3])

# %%
def labels_img_to_color_img(labels: np.ndarray):
    """
        - labels:   Label image, shape (H, W): 0 to N-1 (N labels)
        - returns:  Color image of shape (H, W, 3)
    """
    lb_flat = ein.rearrange(labels, "h w -> (h w)")
    cs = sns.color_palette(n_colors=(int(max(set(lb_flat))) + 1))
    colors = (np.array(cs) * 255).astype(np.uint8)
    colors = {i: colors[i] for i in range(len(colors))}
    colors_img = np.zeros((labels.shape[0], labels.shape[1], 3), 
            dtype=np.uint8)
    for i, rgb in colors.items():
        colors_img[labels == i] = rgb
    return colors_img, cs


# %%
a, cs = labels_img_to_color_img(img_labels[0, 0].numpy())
rimg_np = cv.resize(img_np, (w_new, h_new))
bimg = cv.addWeighted(rimg_np, 0.5, a, 0.5, 0)

# %%
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(a)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(rimg_np)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
plt.figure()
plt.imshow(bimg)
plt.axis('off')
plt.show()


# %%
