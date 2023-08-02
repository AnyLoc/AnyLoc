# Doing VLAD with SAM descriptors
"""
    Basic idea is to extract descriptors from [1] and do VLAD on them.
    
    Note: '--prog.wandb-save-qual' is not used
    
    [1]: https://github.com/facebookresearch/segment-anything
"""

# %%
import os
import sys
from pathlib import Path
# Set the './../' from the script folder
# Also set the './../segment_anything/' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
seg_path = os.path.join(lib_path,'segment-anything')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')

if seg_path not in sys.path:
    print(f'Adding library path: {seg_path} to PYTHONPATH')
    sys.path.append(seg_path)
else:
    print(f'Library path {seg_path} already in PYTHONPATH')

# %%
import torch
from torch.nn import functional as F

from segment_anything import build_sam, SamPredictor, sam_model_registry

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
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens
from custom_datasets.aerial_dataloader import Aerial
import cv2
import pdb
import cv2
import matplotlib.cm as cm
from skimage import color
import torchvision.transforms as T
from matplotlib.lines import Line2D
import imageio.v2 as imageio


def color_map_color(value, cmap_name='jet', vmin=0, vmax=1):
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    return rgb


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-Descs", 
        wandb_group="Direct-Descs")
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # SAM parameters
    model_type: Literal["vit_l","vit_b","vit_h"] = "vit_l"
    """
        Model for SAM to use as the base model.
    """
    # model_path: Literal = "/ocean/projects/cis220039p/jkarhade/data/sam_model/sam_vit_h_4b8939.pth"
    # Number of clusters for VLAD
    num_clusters: int = 8
    # Down-scaling H, W resolution for images (before giving to SAM)
    down_scale_res: Tuple[int, int] = (224, 298)
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Sub-sample query images (RAM or VRAM constraints) (1 = off)
    sub_sample_qu: int = 1
    # Sub-sample database images (RAM or VRAM constraints) (1 = off)
    sub_sample_db: int = 1
    # Sub-sample database images for VLAD clustering only
    sub_sample_db_vlad: int = 1
    """
        Use sub-sampling for creating the VLAD cluster centers. Use
        this to reduce the RAM usage during the clustering process.
        Unlike `sub_sample_qu` and `sub_sample_db`, this is only used
        for clustering and not for the actual VLAD computation.
    """
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Show a matplotlib plot for recalls
    show_plot: bool = False
    # Use hard or soft descriptor assignment for VLAD
    vlad_assignment: Literal["hard", "soft"] = "hard"
    # Softmax temperature for VLAD (soft assignment only)
    vlad_soft_temp: float = 1.0
    # ViT output layer number
    out_layer_num: int = -1
    # ViT use neck
    use_neck:bool = True
    
    qu_indices = None
    qu_in_db = False

# %%
# ---------------- Functions ----------------
@torch.no_grad()
def build_vlads(largs: LocalArgs, vpr_ds: BaseDataset, 
            verbose: bool=True) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        WARNING: docstring is outdated
        
        Build VLAD vectors for database and query images.
        
        Parameters:
        - largs: LocalArgs  Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - verbose: bool     Prints progress if True
        
        Returns:
        - db_vlads:     VLAD descriptors of database of shape 
                        [n_db, vlad_dim]
                        - n_db: Number of database images
                        - vlad_dim: num_clusters * d_dim
                            - d_dim: Descriptor dimensionality
                            - num_clusters: Number of clusters
        - qu_vlads:     VLAD descriptors of queries of shape 
                        [n_qu, vlad_dim], 'n_qu' is num. of queries
    """
    # SAM extractor
    sam = sam_model_registry[largs.model_type](checkpoint= "/ocean/projects/cis220039p/jkarhade/data/sam_model/sam_vit_b_01ec64.pth")
    sam.to(device)
    predictor = SamPredictor(sam,use_neck=largs.use_neck,out_layer_num=largs.out_layer_num)
    
    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    
    # Get cluster centers in the VLAD
    if verbose:
        print("Building VLAD cluster centers...")
    db_indices = np.arange(0, num_db, largs.sub_sample_db_vlad)
    # Database descriptors (for VLAD clusters): [n_db, n_d, d_dim]
    full_db_vlad = []
    # Get global database descriptors
    for i in tqdm(db_indices, disable=(not verbose)):
        img = vpr_ds[i][0]
        # img = ein.rearrange(img, "c h w -> 1 c h w")#.to(device)
        # print(img.shape)
        #F.interpolate(img, largs.down_scale_res)
        # img = img.squeeze(0)

        img = cv2.resize(img,largs.down_scale_res,interpolation = cv2.INTER_AREA)
        predictor.set_image(img)
        desc = predictor.get_image_embedding()

        # import pdb; pdb.set_trace()
        desc = desc.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).unsqueeze(dim=1)
        # print(desc.shape)
        full_db_vlad.append(desc.squeeze().cpu())
    full_db_vlad = torch.stack(full_db_vlad)
    if verbose:
        print(f"Database (for VLAD) shape: {full_db_vlad.shape}")
    d_dim = full_db_vlad.shape[2]
    if verbose:
        print(f"Descriptor dimensionality: {d_dim}")
    vlad = VLAD(largs.num_clusters, d_dim, 
            vlad_mode=largs.vlad_assignment, 
            soft_temp=largs.vlad_soft_temp)
    vlad.fit(ein.rearrange(full_db_vlad, "n k d ->(n k) d"))
    if verbose:
        print(f"VLAD cluster centers shape: {vlad.c_centers.shape}")
    del full_db_vlad
    
    # Get VLADs of the database
    if verbose:
        print("Building VLADs for database...")
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    # All database descriptors (local descriptors): [n_db, n_d, d_dim]
    full_db = []
    # Get global database descriptors
    for i in tqdm(db_indices, disable=(not verbose)):
        img = vpr_ds[i][0]
        # img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
        #F.interpolate(img, largs.down_scale_res)
        # img = img.squeeze(0)

        img = cv2.resize(img,largs.down_scale_res,interpolation = cv2.INTER_AREA)
        predictor.set_image(img)
        desc = predictor.get_image_embedding()
        desc = desc.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).unsqueeze(dim=1)

        full_db.append(desc.squeeze().cpu())
    full_db = torch.stack(full_db)
    if verbose:
        print(f"Full database descriptor shape: {full_db.shape}")
    db_vlads: torch.Tensor = vlad.generate_multi(full_db)
    if verbose:
        print(f"Database VLADs shape: {db_vlads.shape}")
    del full_db
    
    # Get VLADs of the queries
    if verbose:
        print("Building VLADs for queries...")
    qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    full_qu = []
    # Get global descriptors for queries
    for i in tqdm(qu_indices, disable=(not verbose)):
        img = vpr_ds[i][0]
        # img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
        #F.interpolate(img, largs.down_scale_res)
        # img = img.squeeze(0)
        
        img = cv2.resize(img,largs.down_scale_res,interpolation = cv2.INTER_AREA)
        predictor.set_image(img)
        desc = predictor.get_image_embedding()
        desc = desc.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).unsqueeze(dim=1)
        full_qu.append(desc.squeeze().cpu())
    full_qu = torch.stack(full_qu)
    print(full_qu.shape)
    if verbose:
        print(f"Full query descriptor shape: {full_qu.shape}")
    qu_residuals: torch.Tensor = vlad.generate_multi_res_vec(full_qu)
    qu_vlads: torch.Tensor = vlad.generate_multi(full_qu)
    if verbose:
        print(f"Query VLADs shape: {qu_vlads.shape}")
        # print(f"Query Residuals shape: {qu_residuals.shape}")
    del full_qu
    # Return VLADs
    return db_vlads,vlad.c_centers,qu_residuals


@torch.no_grad()
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything(42)
    
    if largs.prog.use_wandb:
        # Launch WandB
        wandb_run = wandb.init(project=largs.prog.wandb_proj, 
                entity=largs.prog.wandb_entity, config=largs,
                group=largs.prog.wandb_group, 
                name=largs.prog.wandb_run_name)
        print(f"Initialized WandB run: {wandb_run.name}")
    
    print("--------- Generating VLADs ---------")
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}, split: {largs.data_split}")
    # Load dataset
    if ds_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(largs.bd_args, ds_dir, ds_name, 
                            largs.data_split,use_SAM=True)
    elif ds_name=="Oxford":
        vpr_ds = Oxford(ds_dir,use_SAM=True)
    elif ds_name=="gardens":
        vpr_ds = Gardens(largs.bd_args,ds_dir,ds_name,largs.data_split,use_SAM=True)
    elif ds_name=="test_40_midref_rot0" or ds_name=="test_40_midref_rot90":
        vpr_ds = Aerial(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split,use_SAM=True)
    # Get VLADs of the database 
    #   From: qu_residuals: [n_qu, 4015=(55*73), n_c, d_dim]
    db_vlads, vlad_cluster_centers, qu_residuals = build_vlads(largs, 
                                                    vpr_ds)
    print("--------- Generated VLADs ---------")
    
    print("---------- Visualizing Cluster Centers Assignment ---------")
    # Visualize VLAD clusters
    
    colors = np.zeros((largs.num_clusters,3))
    legend_lines = []
    legend_nums = []
    for j in range(largs.num_clusters):
        colors[j,:] = color_map_color(j/(largs.num_clusters-1))
        custom_line = Line2D([0], [0], color = color_map_color(
                j/(largs.num_clusters-1)), lw=4)
        legend_lines.append(custom_line)
        legend_nums.append(str(j))
    
    # print(f"Colors: {colors}")
    # Ensure that save directory exists
    save_fldr = largs.prog.cache_dir
    if largs.exp_id is not None:
        save_fldr = f"{save_fldr}/experiments/{largs.exp_id}"
    save_fldr = f"{save_fldr}/vlad_clusters_viz"
    save_fldr = os.path.realpath(os.path.expanduser(save_fldr))
    if not os.path.exists(save_fldr):
        os.makedirs(save_fldr)
    else:
        print(f"WARNING: Folder {save_fldr} exists! overwriting...")
    print(f"Saving images in: {save_fldr}")
    
    # Loop through all query images
    for i in tqdm(range(qu_residuals.shape[0])):
        # img_orig = cv2.imread(vpr_ds.q_abs_paths[i])
        # img_orig = cv2.resize(img_orig,(73,55))
        if largs.qu_indices is None:
            qi_ds = vpr_ds.database_num + i*largs.sub_sample_qu
        else:
            if largs.qu_in_db:
                qi_ds = largs.qu_indices[i]
            else:
                qi_ds = vpr_ds.database_num + largs.qu_indices[i]
        print(qi_ds)
        img_orig = vpr_ds[qi_ds][0]#.detach().cpu().numpy()
        img_orig = cv2.resize(img_orig,(64,64),interpolation = cv2.INTER_AREA)

        # img_orig = np.asarray(img_orig,dtype=np.uint8)*255
        img_desc = []
        # Loop through all the patches inside image (Dino patches)
        for j in range(qu_residuals.shape[1]):
            cur_res_vec = torch.abs(qu_residuals[i][j])
            res_idx = torch.argmin(torch.sum(cur_res_vec, dim=1))
            img_desc.append(res_idx)
        img_desc = np.reshape(np.asarray(img_desc),(64,64))
        # Color based on the closest clusters
        all_color_img = np.zeros_like(img_orig) # [55, 73, 3]
        for c in range(largs.num_clusters):
            img_idx = (np.argwhere(img_desc==c))
            all_color_img[img_idx[:,0],img_idx[:,1],:] = colors[c]*255
        # Merge cluster color map with original image
        img_original = vpr_ds[qi_ds][0]
        sz = tuple(img_original.shape[:2]) # [H, W]
        # img_original = cv2.resize(img_original,(64,64),interpolation = cv2.INTER_AREA)
        all_color_img_resized = cv2.resize(all_color_img, sz[::-1], 
                interpolation=cv2.INTER_NEAREST)
        # all_color_img_resized = all_color_img
        # color_layer_img = cv2.addWeighted(img_orig, 0.7, 
        #         all_color_img, 0.3, 0)
        print(img_original.shape,all_color_img_resized.shape,sz)
        color_layer_img = cv2.addWeighted(img_original, 0.7,
                all_color_img_resized, 0.3, 0)
        # color_layer_img = cv2.resize(color_layer_img,(298,224))
        color_layer_img = color_layer_img/255   # [0 - 1] float range
        # color_layer_img = np.asarray(color_layer_img,dtype=np.uint8)
        # Plot and save
        fig,ax = plt.subplots()
        splits = vpr_ds.get_image_paths()[qi_ds].split("/")[-2:]
        f_title = str(os.path.join(*splits))
        ax.set_title(f_title)
        im = ax.imshow(color_layer_img)
        if largs.num_clusters <= 16:
            ax.legend(legend_lines, legend_nums, loc='upper left', 
                    bbox_to_anchor=(1.025, 1.05))
        else:
            # cb = plt.colorbar(im)
            # TODO: Want a custom colorbar with color labels?
            pass
        # plt.show()
        fig.set_tight_layout(True)
        # print(f"{save_fldr}/{i}.png")
        fig.savefig(f"{save_fldr}/{i}.png")
        plt.close()
    
    # If saving GIF
    frames = []
    if largs.save_gif:
        for i in range(qu_residuals.shape[0]):
            frames.append(imageio.imread(f"{save_fldr}/{i}.png"))
        ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
        imageio.mimsave(f"{save_fldr}/{ts}.gif", frames, fps=2)
    
    print("----- Finished visualization of cluster centers -----")


# %%
if __name__ == "__main__" and ("ipykernel" not in sys.argv[0]):
    largs = tyro.cli(LocalArgs, description=__doc__)
    _start = time.time()
    try:
        main(largs)
    except:
        print("Unhandled exception")
        traceback.print_exc()
    finally:
        print(f"Program ended in {time.time()-_start:.3f} seconds")
        exit(0)
