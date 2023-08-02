# Cluster image descriptors for Dino-v2 across all datasets
"""
    For all datasets, select representative images. Extract Dino-v2
    image descriptors for each dataset. Project them to 2D using tSNE
    and visualize the descriptors (with color codes for datasets).
    
    Looking for global image descriptors?
    Use the `dino_v2_datasets_gem_tsne_clustering.py` file instead.
    This one treats it as a collection of patches and sub-samples (the
    property of image is not present).
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
import numpy as np
from torchvision.transforms import functional as T
import einops as ein
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tyro
import joblib
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Literal, Union, Tuple, List
import traceback
import time
# Program utilities
from utilities import DinoV2ExtractFeatures, CustomDataset, \
        seed_everything
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from custom_datasets.global_dataloader import Global_Dataloader \
        as GlobalDataloader
from dvgl_benchmark.datasets_ws import BaseDataset
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens
from custom_datasets.aerial_dataloader import Aerial
from custom_datasets.hawkins_dataloader import Hawkins
from custom_datasets.vpair_dataloader import VPAir
from custom_datasets.laurel_dataloader import Laurel
from custom_datasets.eiffel_dataloader import Eiffel
from custom_datasets.vpair_distractor_dataloader import VPAir_Distractor


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(use_wandb=False)
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Dino-v2 parameters
    # Dino-v2 model type
    model_type: Literal["dinov2_vits14", "dinov2_vitb14", 
            "dinov2_vitl14", "dinov2_vitg14"] = "dinov2_vits14"
    # Layer for extracting Dino feature (descriptors)
    desc_layer: int = 11
    # Facet for extracting descriptors
    desc_facet: Literal["query", "key", "value", "token"] = "key"
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Number of images for each dataset (0 = don't use)
    num_images: dict = field(default_factory=lambda: {
        # Database name: number of images
        "Oxford": 50,
        "gardens": 50,
        "17places": 50,
        "baidu_datasets": 50,   # 10
        "st_lucia": 50,    # 10
        "pitts30k": 50,    # 15
        "Tartan_GNSS_test_rotated": 50,
        "Tartan_GNSS_test_notrotated": 50,
        "hawkins": 50,
        "laurel_caverns": 50,
        "eiffel": 50,
        "VPAir": 50
    })
    # Force inference image (H, W) size (bicubic interpolation)
    img_res: Tuple[int, int] = (480, 640)
    """
        Force a particular image size (so that the image sizes in the
        dataset doesn't bias the result). Note that, if the numbers 
        aren't divisible by 14, the final size won't be this but its 
        nearest (lower) multiple of 14 (the patch size for Dino-v2).
    """
    # Sub-sampling the final tSNE figure
    subsample_final: float = 0.001
    """
        If 0.2, then only 20% of the descriptors from each dataset are
        actually visualized. Use this to reduce clutter in plot.
        Note that for each dataset, the number of descriptors is
        `num_imgs * h * w` (so downsample accordingly). Use this 
        setting with `num_images` to handle representation in final
        plot.
    """
    # Cache the detection results (as joblib dump)
    cache_fname: Union[str, None] = None
    """
        This should be a file in the cache directory. It can contain
        '/' for folders. Don't use extensions.
        
        - If an existing file, then it's used to read the dump (and
            restore cache). No inference happens then.
        - If not an existing file, then it's used to save the dump.
        - If None, then no caching is done.
        
        If caching is used, the following is cached
        - Result cache for all image descriptors across all datasets.
            The file extension is '.gz'
        - tSNE projections of the descriptors (you only need this for
            plotting)
            The file ends with '_tsne.gz'
    """
    # Show the plot (False = No plot but save figure, True = Show fig)
    show_plot: bool = True


# %%
@torch.no_grad()
def build_cache(largs: LocalArgs, verbose: bool=True):
    """
        Builds the cache dictionary.
        
        Cache dictionary format:
        - "time": str       Time stamp of creation
        - "model": dict
            - "type":   `largs.model_type`
            - "layer":  `largs.desc_layer`
            - "facet":  `largs.desc_facet`
        - "data": dict      Data dictionary
            - "<Dataset Name>": dict
                - "indices": List[int]
                    Indices for the images chosen (at random)
                - "descriptors": np.ndarray
                    Image descriptors of shape [N, n_d, d_dim] where
                    N is number of images, n_d is h * w (number of
                    14x14 patches per image), d_dim is the descriptor
                    dimension.
    """
    # Create Dino-v2 model
    dino = DinoV2ExtractFeatures(largs.model_type, largs.desc_layer,
            largs.desc_facet, device=device)
    res = {
        "model": {
            "type": largs.model_type,
            "layer": largs.desc_layer,
            "facet": largs.desc_facet
        },
        "data": {}
    }
    
    # Extracts descriptors for the given indices from given dataset
    def extract_patch_descs(vpr_ds: CustomDataset, 
                            indices: List[int]):
        patch_descs = []
        for i in tqdm(indices, disable=(not verbose)):
            img = vpr_ds[i][0].to(device)
            img = T.resize(img, largs.img_res)
            c, h, w = img.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_in = T.center_crop(img, (h_new, w_new))[None, ...]
            res = dino(img_in)
            patch_descs.append(res.cpu())
        patch_descs = torch.cat(patch_descs, dim=0)
        return patch_descs
    
    ds_dir = largs.prog.data_vg_dir
    ds_split = largs.data_split
    bd_args = largs.bd_args
    print(f"Dataset directory: {ds_dir}, split: {ds_split}")
    ds_names = [k for k in largs.num_images \
                    if largs.num_images[k] > 0]
    # for _ in tqdm(range(100), leave=False, position=0, desc="DB"):
    for ds_name in tqdm(ds_names, disable=(not verbose)):
        # Load dataset
        if ds_name=="baidu_datasets":
            vpr_ds = Baidu_Dataset(bd_args, ds_dir, ds_name, ds_split)
        elif ds_name=="Oxford":
            vpr_ds = Oxford(ds_dir)
        elif ds_name=="gardens":
            vpr_ds = Gardens(bd_args, ds_dir, ds_name, ds_split)
        elif ds_name.startswith("Tartan_GNSS"):
            vpr_ds = Aerial(bd_args, ds_dir, ds_name, ds_split)
        elif ds_name.startswith("hawkins"): # Use only long_corridor
            vpr_ds = Hawkins(bd_args, ds_dir,"hawkins_long_corridor", 
                    ds_split)
        elif ds_name=="VPAir":
            vpr_ds = VPAir(bd_args, ds_dir, ds_name, ds_split)
            vpr_distractor_ds = VPAir_Distractor(bd_args, ds_dir, 
                    ds_name, ds_split)
        elif ds_name=="laurel_caverns":
            vpr_ds = Laurel(bd_args, ds_dir, ds_name, ds_split)
        elif ds_name=="eiffel":
            vpr_ds = Eiffel(bd_args, ds_dir, ds_name, ds_split)
        else:
            vpr_ds = BaseDataset(bd_args, ds_dir, ds_name, ds_split)
        # Random indices
        db_repr_samples: np.ndarray = np.random.choice(
                np.arange(len(vpr_ds)), largs.num_images[ds_name],
                replace=False)
        # Get patch descriptors [N, n_d, d_dim]
        patch_descs = extract_patch_descs(vpr_ds, db_repr_samples)
        # Add to result
        res["data"][ds_name] = {
            "indices": db_repr_samples.tolist(),
            "descriptors": patch_descs.numpy()
        }
    # Add timestamp
    res["time"] = str(time.strftime(f"%Y_%m_%d_%H_%M_%S"))
    return res


# %%
# From the descriptors (from cache) build tSNE projections
def tsne_project(largs: LocalArgs, res: dict):
    descs_all = []
    labels_db = {}
    for ds_name in res["data"]:
        descs = res["data"][ds_name]["descriptors"]
        descs = ein.rearrange(descs, "n n_d d -> (n n_d) d")
        d = descs.shape[0]
        i = np.random.choice(np.arange(d), 
                int(d * largs.subsample_final), replace=False)
        descs = descs[i]
        labels_db[ds_name] = descs.shape[0]
        descs_all.append(descs)
    descs_all = np.concatenate(descs_all, axis=0)
    # tSNE projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=25)
    desc_2d = tsne.fit_transform(descs_all)
    # Convert back to dictionary
    descs_db = {}
    i = 0
    for db in labels_db:
        descs_db[db] = desc_2d[i:i+labels_db[db], :]
        i += labels_db[db]
    return descs_db


# %%
def plot_tsne(largs: LocalArgs, descs_db: dict):
    # Colors and markers
    db_colors = {
        "Oxford": "#008000",
        "gardens": "#004ccc",
        "17places": "#004ccc",
        "baidu_datasets": "#004ccc",
        "st_lucia": "#008000",
        "pitts30k": "#008000",
        "Tartan_GNSS_test_rotated": "#800080",
        "Tartan_GNSS_test_notrotated": "#800080",
        "hawkins": "#80471c",
        "laurel_caverns": "#80471c",
        "eiffel": "#297bd8",
        "VPAir": "#800080"
    }
    db_markers = {
        "Oxford": "^",
        "gardens": "p",
        "17places": "P",
        "baidu_datasets": "*",
        "st_lucia": "v",
        "pitts30k": "<",
        "Tartan_GNSS_test_rotated": "_",
        "Tartan_GNSS_test_notrotated": "|",
        "hawkins": "1",
        "laurel_caverns": "2",
        "eiffel": "x",
        "VPAir": "d"
    }
    # Plot figure
    plt.figure()
    for db in descs_db:
        plt.scatter(descs_db[db][:, 0], descs_db[db][:, 1], label=db, 
                c=db_colors[db], marker=db_markers[db])
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if largs.show_plot:
        plt.show()
    else:
        cache_fname = os.path.realpath(os.path.expanduser(
            os.path.join(largs.prog.cache_dir, largs.cache_fname)))
        plt.savefig(f"{cache_fname}_tsne.png")


# %%
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything(42)
    
    # Check if cache files exists
    use_caching, cache_fname = False, None
    if largs.prog.cache_dir is not None and \
            largs.cache_fname is not None:
        use_caching = True
        cache_fname = os.path.realpath(os.path.expanduser(
            os.path.join(largs.prog.cache_dir, largs.cache_fname)))
        print(f"Using cache file: {cache_fname}")
        d_name = os.path.dirname(cache_fname)
        if os.path.isdir(d_name):
            print(f"Directory {d_name} exists! (could overwrite)")
        else:
            os.makedirs(d_name)
            print(f"Created directory {d_name}")
    else:
        print(f"Not using any caching (results won't be saved)")
    
    # Get the descriptors
    if use_caching and os.path.exists(f"{cache_fname}.gz"):
        print(f"Loading from cache file: {cache_fname}.gz")
        res = joblib.load(f"{cache_fname}.gz")
    else:
        res = build_cache(largs)
        if use_caching:
            print(f"Saving to cache file: {cache_fname}.gz")
            joblib.dump(res, f"{cache_fname}.gz")
    
    # tSNE
    if use_caching and os.path.exists(f"{cache_fname}_tsne.gz"):
        print(f"Loading from cache file: {cache_fname}_tsne.gz")
        descs_db = joblib.load(f"{cache_fname}_tsne.gz")
    else:
        descs_db = tsne_project(largs, res)
        if use_caching:
            print(f"Saving to cache file: {cache_fname}_tsne.gz")
    
    # Plot
    plot_tsne(largs, descs_db)


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
# Experimental section

# %%
ds_dir = "/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
ds_split = "test"
bd_args = base_dataset_args
print(f"Dataset directory: {ds_dir}")

# %%
model_type: Literal["dinov2_vits14", "dinov2_vitb14", 
            "dinov2_vitl14", "dinov2_vitg14"] = "dinov2_vitg14"
desc_layer: int = 31
desc_facet: Literal["query", "key", "value", "token"] = "value"

# %%
largs = LocalArgs(ProgArgs(data_vg_dir=ds_dir), model_type=model_type,
        desc_layer=desc_layer, desc_facet=desc_facet)

# %%
res = build_cache(largs)

# %%
