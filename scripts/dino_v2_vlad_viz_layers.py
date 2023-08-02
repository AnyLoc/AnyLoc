# Doing VLAD with DINOv2 descriptors and visualizing the clusters
"""
    Get the VLAD vocabulary (cluster centers) for DINOv2 descriptors
    (ViT features from particular layers), get the residuals for each
    descriptor, get closest cluster center (argmin distance) and
    visualize it as an overlay.
    Basically shows the cluster number each (patch) descriptor belongs
    (VLAD assignment visualization).
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
from torch.nn import functional as F
from utilities import DinoV2ExtractFeatures
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
import distinctipy
from tqdm.auto import tqdm
from dvgl_benchmark.datasets_ws import BaseDataset
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from typing import Union, Literal, Tuple, List
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens
from custom_datasets.hawkins_dataloader import Hawkins
from custom_datasets.aerial_dataloader import Aerial
from custom_datasets.vpair_dataloader import VPAir
from custom_datasets.laurel_dataloader import Laurel
from custom_datasets.eiffel_dataloader import Eiffel
from custom_datasets.vpair_distractor_dataloader import VPAir_Distractor
import pdb
import cv2
import matplotlib.cm as cm
from skimage import color
import torchvision.transforms as T
from matplotlib.lines import Line2D
import imageio.v2 as imageio

device = torch.device("cuda:0")


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-Descs", 
        wandb_group="viz_vlad_clusters", use_wandb=False)
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # Dino parameters
    model_type: Literal["dinov2_vits14", "dinov2_vitb14", 
            "dinov2_vitl14", "dinov2_vitg14"] = "dinov2_vits14"
    """
        Model for Dino-v2 to use as the base model.
    """
    # Number of clusters for VLAD
    num_clusters: int = 32
    # Layers for extracting Dino feature (descriptors)
    desc_layers: List[int] = field(default_factory=lambda:\
                                list(range(0, 40, 1)))
    # Facet for extracting descriptors
    desc_facet: Literal["query", "key", "value", "token"] = "token"
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
    # Override the query indices (None = don't use override)
    qu_indices: Union[List[int], None] = None
    # Override queries to be placed in database images
    qu_in_db: bool = False
    """
        If True, the 'qu_indices' are treated in the database set and
        not the query set. This setting is valid only if 'qu_indices'
        is not None.
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
    # Caching configuration
    cache_vlad_descs: bool = False
    # Save the resultant images as a GIF (filename is timestamp)
    save_gif: bool = False
    # Override the default cluster cache directory (if not None)
    override_vlad_cdir: Union[str, None] = None
    # Random number for distinctipy_color
    distinctipy_rng: int = 928


# %%
# ---------------- Functions ----------------
# jet, nipy_spectral_r, tab20
def color_map_color(value, cmap_name='jet', vmin=0, vmax=1):
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)
    print(value)
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    return rgb


def color_map_distinctipy_color(c_val, num_c, rng=928):
    """
        Generate colors for each cluster
        - c_val: Cluster number (0 to num_c-1)
        - num_c: Number of clusters
        - rng: Random seed
        
        Returns:
        - (r, g, b) value
    """
    colors = distinctipy.get_colors(num_c, rng=rng, 
            colorblind_type="Deuteranomaly")
    return colors[c_val]

@torch.no_grad()
def build_vlads(largs: LocalArgs, vpr_ds: BaseDataset, 
            dino: DinoV2ExtractFeatures, verbose: bool=True, 
            dino_layer: int=11) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Build VLAD vectors for database and query images.
        
        Parameters:
        - largs: LocalArgs      Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - dino:     The DINO-v2 model
        - verbose: bool     Prints progress if True
        - dino_layer: int   Dino layer to extract descriptors from
        
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
    cache_dir = None
    if largs.cache_vlad_descs:
        if largs.override_vlad_cdir is not None:
            cache_dir = largs.override_vlad_cdir
        else:
            cache_dir = f"{largs.prog.cache_dir}/vlad_descs/Dinov2/" \
                        f"{largs.prog.vg_dataset_name}/" \
                        f"{largs.model_type}-{largs.desc_facet}"
            cache_dir += f"-L{dino_layer}-C{largs.num_clusters}"
        print(f"Using cache directory: {cache_dir}")
    
    vlad = VLAD(largs.num_clusters, None, 
            vlad_mode=largs.vlad_assignment, 
            soft_temp=largs.vlad_soft_temp, cache_dir=cache_dir)
    
    def extract_patch_descriptors(indices):
        patch_descs = []
        for i in tqdm(indices, disable=(not verbose)):
            img = vpr_ds[i][0].to(device)
            c, h, w = img.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_in = T.CenterCrop((h_new, w_new))(img)[None, ...]
            ret = dino(img_in)
            patch_descs.append(ret.cpu())
        patch_descs = torch.cat(patch_descs, dim=0) # [N, n_p, d_dim]
        return patch_descs
    
    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    
    if vlad.can_use_cache_vlad():
        if verbose:
            print("Valid cache found, using it")
        vlad.fit(None)  # Nothing to fit (restore cache)
    else:
        # Get cluster centers in the VLAD
        if verbose:
            print("Building VLAD cluster centers...")
        db_indices = np.arange(0, num_db, largs.sub_sample_db_vlad)
        # Database descriptors (for VLAD clusters): [n_db, n_d, d_dim]
        full_db_vlad = extract_patch_descriptors(db_indices)
        if verbose:
            print(f"Database (for VLAD) shape: {full_db_vlad.shape}")
        d_dim = full_db_vlad.shape[2]
        if verbose:
            print(f"Descriptor dimensionality: {d_dim}")
        vlad.fit(ein.rearrange(full_db_vlad, "n k d -> (n k) d"))
        del full_db_vlad
    # VLAD cluster centers loaded
    if verbose:
        print(f"VLAD cluster centers shape: "\
                f"{vlad.c_centers.shape}, ({vlad.c_centers.dtype})")
    
    db_vlads = None     # TODO: Not using for now. Do not remove yet.
    # # Get VLADs of the database
    # if verbose:
    #     print("Building VLADs for database...")
    # db_indices = np.arange(0, num_db, largs.sub_sample_db)
    # db_img_names = vpr_ds.get_image_relpaths(db_indices)
    # if vlad.can_use_cache_ids(db_img_names):
    #     if verbose:
    #         print("Valid cache found, using it")
    #     db_vlads = vlad.generate_multi([None] * len(db_indices), 
    #             db_img_names)
    # else:
    #     if verbose:
    #         print("Valid cache not found, doing forward pass")
    #     # All database descs (local descriptors): [n_db, n_d, d_dim]
    #     full_db = extract_patch_descriptors(db_indices)
    #     if verbose:
    #         print(f"Full database descriptor shape: {full_db.shape}")
    #     db_vlads: torch.Tensor = vlad.generate_multi(full_db, 
    #             db_img_names)
    #     del full_db
    # if verbose:
    #     print(f"Database VLADs shape: {db_vlads.shape}")
    
    # Get VLADs of the queries
    if verbose:
        print("Building VLADs for queries...")
    if largs.qu_indices is None:
        qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    else:
        if largs.qu_in_db:
            qu_indices = np.array(largs.qu_indices)
        else:
            qu_indices = np.array(largs.qu_indices) + num_db
    qu_img_names = vpr_ds.get_image_relpaths(qu_indices)
    if vlad.can_use_cache_ids(qu_img_names, only_residuals=True):
        if verbose:
            print("Valid cache found, using it")
        qu_residuals: torch.Tensor = vlad.generate_multi_res_vec(
                [None] * len(qu_indices), qu_img_names)
    else:
        full_qu = extract_patch_descriptors(qu_indices)
        if verbose:
            print(f"Full query descriptor shape: {full_qu.shape}")
        qu_residuals: torch.Tensor = vlad.generate_multi_res_vec(
                full_qu, qu_img_names)
        del full_qu
    
    if verbose:
        print(f"Query Residuals shape: {qu_residuals.shape}")
    # Return VLADs
    return db_vlads, vlad.c_centers, qu_residuals


# %%
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
    
    print("--------- Initializing dataset ---------")
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}, split: {largs.data_split}")
    # Load dataset
    if ds_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(largs.bd_args, ds_dir, ds_name, 
                            largs.data_split)
    elif ds_name=="Oxford":
        vpr_ds = Oxford(ds_dir)
    elif ds_name=="gardens":
        vpr_ds = Gardens(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name.startswith("Tartan_GNSS"):
        vpr_ds = Aerial(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name.startswith("hawkins"): # Use only long_corridor
        vpr_ds = Hawkins(largs.bd_args,ds_dir,"hawkins_long_corridor",largs.data_split)
    elif ds_name=="VPAir":
        vpr_ds = VPAir(largs.bd_args,ds_dir,ds_name,largs.data_split)
        vpr_distractor_ds = VPAir_Distractor(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="laurel_caverns":
        vpr_ds = Laurel(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="eiffel":
        vpr_ds = Eiffel(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split)
    print("--------- Dataset initialized ---------")
    
    print("---------- Visualizing Cluster Centers Assignment ---------")
    # Visualize VLAD clusters
    
    colors = np.zeros((largs.num_clusters,3))
    legend_lines = []
    legend_nums = []
    use_distinctipy = True
    for j in range(largs.num_clusters):
        if not use_distinctipy:
            cval = color_map_color(j/(largs.num_clusters-1))
        else:
            cval = color_map_distinctipy_color(j, largs.num_clusters,
                    rng=largs.distinctipy_rng)
        colors[j,:] = cval
        custom_line = Line2D([0], [0], color = cval, lw=4)
        legend_lines.append(custom_line)
        legend_nums.append(str(j))
    
    # print(f"Colors: {colors}")
    # Ensure that save directory exists
    save_fldr = largs.prog.cache_dir
    if largs.exp_id is not None:
        save_fldr = f"{save_fldr}/experiments/{largs.exp_id}"
    save_fldr = f"{save_fldr}/vlad_clusters_viz_dinov2_g"
    save_fldr = os.path.realpath(os.path.expanduser(save_fldr))
    if not os.path.exists(save_fldr):
        os.makedirs(save_fldr)
    else:
        print(f"WARNING: Folder {save_fldr} exists! overwriting...")
    print(f"Saving images in: {save_fldr}")
    
    for l in largs.desc_layers:
        # Get VLADs of the database 
        #   From: qu_residuals: [n_qu, 4015=(55*73), n_c, d_dim]
        print(f"--- Layer {l} ---")
        # Dino feature extraction model
        # dino = None # TODO: FIXME: Hack to read faster from cache
        dino = DinoV2ExtractFeatures(largs.model_type, l,
                largs.desc_facet, device=device)
        print("Dino model created")
        db_vlads, vlad_cluster_centers, qu_residuals = build_vlads(
                largs, vpr_ds, dino, dino_layer=l)
        del dino
        qu_residuals = qu_residuals.cpu()
        print("Dino model deleted")
        
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
            img_orig = vpr_ds[qi_ds][0]#.detach().cpu().numpy()
            _c, h, w = img_orig.shape
            _h, _w = (h // 14), (w // 14)  # Not necessarily (55, 73)!
            resize_T = T.Resize((_h,_w))   # Dino patches
            img_orig = resize_T(img_orig)
            # ImageNet normalization
            mu = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1)\
                .unsqueeze(-1)
            std= torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1)\
                .unsqueeze(-1)
            # print(img_orig.shape,std.shape,mu.shape)
            img_orig = img_orig * std + mu
            img_orig = img_orig.detach().cpu().numpy()
            img_orig = np.moveaxis(img_orig, 0, -1)*255   # C last
            # img_orig = np.asarray(img_orig,dtype=np.uint8)*255
            img_desc = []
            # Loop through all the patches inside image (Dino patches)
            for j in range(qu_residuals.shape[1]):
                cur_res_vec = torch.abs(qu_residuals[i][j])
                res_idx = torch.argmin(torch.sum(cur_res_vec, dim=1))
                img_desc.append(res_idx)
            img_desc = np.reshape(np.asarray(img_desc),(_h,_w))
            # Color based on the closest clusters
            all_color_img = np.zeros_like(img_orig) # [55, 73, 3]
            for c in range(largs.num_clusters):
                img_idx = (np.argwhere(img_desc==c))
                all_color_img[img_idx[:,0],img_idx[:,1],:] = colors[c]*255
            # Merge cluster color map with original image
            img_original = vpr_ds[qi_ds][0]
            sz = tuple(img_original.shape[1:]) # [H, W]
            resize_T = T.Resize(sz)    # Original shape ;)
            img_original = resize_T(img_original)
            img_original = img_original * std + mu
            img_original = img_original.detach().cpu().numpy()
            img_original = np.moveaxis(img_original, 0, -1)*255
            all_color_img_resized = cv2.resize(all_color_img, sz[::-1], 
                    interpolation=cv2.INTER_NEAREST)
            # color_layer_img = cv2.addWeighted(img_orig, 0.7, 
            #         all_color_img, 0.3, 0)
            color_layer_img = cv2.addWeighted(img_original, 0.7,
                    all_color_img_resized, 0.3, 0)
            # color_layer_img = cv2.resize(color_layer_img,(298,224))
            color_layer_img = color_layer_img/255   # [0 - 1] float range
            # color_layer_img = np.asarray(color_layer_img,dtype=np.uint8)
            # Plot and save
            font_dict = {'size': 14, 'fontweight': 'bold'}
            fig, ax = plt.subplots()
            splits = vpr_ds.get_image_paths()[qi_ds].split("/")[-2:]
            f_title = str(os.path.join(*splits))
            # ax.set_title(f_title)
            _i = str(largs.qu_indices[i])
            if largs.qu_in_db:
                ax.set_title(f"Layer: {l} - Image: {_i} (DB)", 
                            fontdict=font_dict)
            else:
                ax.set_title(f"Layer: {l} - Image: {_i} (QU)",
                            fontdict=font_dict)
            im = ax.imshow(color_layer_img)
            # if largs.num_clusters <= 16:
            #     ax.legend(legend_lines, legend_nums, loc='upper left', 
            #             bbox_to_anchor=(1.025, 1.05))
            # else:
            #     # cb = plt.colorbar(im)
            #     # TODO: Want a custom colorbar with color labels?
            #     pass
            # plt.show()
            ax.axis('off')
            fig.set_tight_layout(True)
            fig.savefig(f"{save_fldr}/L_{l}_I_{i}.png")
            plt.close()
            # Save as RGB
            # color_layer_img = (color_layer_img * 255).astype(np.uint8)
            # Image.fromarray(color_layer_img, mode="RGB")\
            #         .save(f"{save_fldr}/L_{l}_I_{i}.png")
    
    # If saving GIF
    if largs.save_gif:
        print(f"Saving GIFs for all images in: {save_fldr}")
        # Save across all layers in each image
        # for i in range(qu_residuals.shape[0]):
        #     frames = []
        #     for l in largs.desc_layers:
        #         frames.append(imageio.imread(
        #                 f"{save_fldr}/L_{l}_I_{i}.png"))
        #     ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
        #     imageio.mimsave(f"{save_fldr}/I_{i}_{ts}.gif", frames, 
        #             fps=1)
        # Save across all images in each layer
        for l in largs.desc_layers:
            frames = []
            for i in range(qu_residuals.shape[0]):
                frames.append(imageio.imread(
                        f"{save_fldr}/L_{l}_I_{i}.png"))
            ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
            imageio.mimsave(f"{save_fldr}/L_{l}_{ts}.gif", frames, 
                    fps=5)
    
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

# %%
