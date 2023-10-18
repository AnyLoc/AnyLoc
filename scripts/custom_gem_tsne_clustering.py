# Cluster image descriptors for custom algorithms
"""
    Given a folder with 'npy' files (for descriptors), sub-sample,
    do tSNE, and show projections.
    The 'npy' files are generated for CosPlace, MixVPR, and NetVLAD.
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
import einops as ein
from typing import Literal, Union, Tuple, List
from dataclasses import dataclass, field
import joblib
import tyro
import traceback
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from utilities import seed_everything
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
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Number of images for each dataset (0 = don't use)
    num_images: dict = field(default_factory=lambda: {
        # Database name: number of images
        "Oxford": 10,
        "gardens": 10,
        "17places": 10,
        "baidu_datasets": 10,
        "st_lucia": 10,
        "pitts30k": 10,
        "Tartan_GNSS_test_rotated": 10,
        "Tartan_GNSS_test_notrotated": 10,
        "hawkins": 10,
        "laurel_caverns": 10,
        "eiffel": 10,
        "VPAir": 10 
    })
    # A multiplier to the number of images (for scaling num of imgs)
    num_imgs_scaling: int = 1
    # Select the image from particular segment
    seg_select: Literal["db", "query", "both"] = "both"
    # Sub-sampling the final tSNE figure
    subsample_final: float = 1.0
    """
        If 0.2, then only 20% of the descriptors from each dataset are
        actually visualized. Use this to reduce clutter in plot.
        Note that for each dataset, the number of descriptors is
        `num_imgs` (so downsample accordingly). Use this setting with 
        `num_images` to handle representation in final plot.
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
    # GeM Pooling Parameter
    gem_p: float = 3
    # Configure the behavior of GeM
    gem_use_abs: bool = False
    """
        If True, the `abs` is applied to the patch descriptors (all
        values are strictly positive). Otherwise, a gimmick involving
        complex numbers is used. If False, the `gem_p` should be an
        integer (fractional will give complex numbers when applied to
        negative numbers as power).
    """
    # Do GeM element-by-element (only if gem_use_abs = False)
    gem_elem_by_elem: bool = False
    """
        Do the GeM element-by-element (only if `gem_use_abs` = False).
        This can be done to prevent the RAM use from exploding for 
        large datasets.
    """
    # Show the plot (False = No plot but save fig, True = show fig)
    show_plot: bool = True
    # Directory where the npy descriptors are stored
    desc_dir: Path = "./data"
    # If True, the database are with high alpha (bold) - plotting only
    fit_db_tf_qu: bool = False
    """
        If True, then after the tSNE projection, the database 
        descriptors are with alpha = 1.0 (opaque/bold) and query
        descriptors are with alpha = 0.5 (semi-transparent).
        Note that both database and query are used in the tSNE 
        projection (as tSNE is a non-parameteric projection). This
        argument is largely to maintain consistency with the PCA 
        codebase.
        Note: seg_select = "both" if this is True.
    """


# %%
def load_descs(largs: LocalArgs, verbose: bool=True):
    # Specific datasets
    d_dir = os.path.realpath(os.path.expanduser(largs.desc_dir))
    assert os.path.isdir(d_dir), f"Invalid desc_dir: {d_dir}"
    ds_names = [k for k in largs.num_images \
                    if largs.num_images[k] > 0]
    res = {
        "data": {}
    }
    ds_dir = largs.prog.data_vg_dir
    ds_split = largs.data_split
    bd_args = largs.bd_args
    print(f"Dataset directory: {ds_dir}, split: {ds_split}")
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
        if largs.seg_select == "both":
            seg_range = np.arange(len(vpr_ds))
        elif largs.seg_select == "db":
            seg_range = np.arange(0, vpr_ds.database_num)
        elif largs.seg_select == "query":
            seg_range = np.arange(vpr_ds.database_num, len(vpr_ds))
        db_repr_samples: np.ndarray = np.random.choice(seg_range, 
                largs.num_images[ds_name] * largs.num_imgs_scaling, 
                replace=False)
        d_np = np.load(f"{d_dir}/{ds_name}.npy")
        d_np = d_np[db_repr_samples]
        res["data"][ds_name] = {
            "indices": db_repr_samples.tolist(),
            "descriptors": d_np,
            "num_db": int(vpr_ds.database_num),
        }
    # Add timestamp
    res["time"] = str(time.strftime(f"%Y_%m_%d_%H_%M_%S"))
    # Return
    return res


# %%
# From the descriptors (from cache) build tSNE projections
def tsne_project(largs: LocalArgs, res: dict):
    descs_all = []
    labels_db = {}
    for ds_name in res["data"]:
        descs = res["data"][ds_name]["descriptors"] # [N, d_dim]
        d = descs.shape[0]
        i = np.random.choice(np.arange(d), 
                int(d * largs.subsample_final), replace=False)
        descs = descs[i]
        labels_db[ds_name] = descs.shape[0]
        descs_all.append(descs)
    descs_all = np.concatenate(descs_all, axis=0)
    # tSNE projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    desc_2d = tsne.fit_transform(descs_all)
    # Convert back to dictionary
    descs_db = {}
    i = 0
    for db in labels_db:
        descs_db[db] = desc_2d[i:i+labels_db[db], :]
        i += labels_db[db]
    # TODO: Add code here to deal with database and query split
    if largs.fit_db_tf_qu:
        # Filter based on index
        res_descs_db = {
            "database": {},
            "queries": {}
        }
        for db in descs_db:
            # All indices and tSNE points
            db_inds = np.array(res["data"][db]["indices"])
            db_tsne = descs_db[db]  # [(n_db + n_qu), 2] t-SNE points
            # Number of database images
            num_db = res["data"][db]["num_db"]
            db_is = db_inds < num_db
            qu_is = np.logical_not(db_is)
            res_descs_db["database"][db] = db_tsne[db_is, :]
            res_descs_db["queries"][db] = db_tsne[qu_is, :]
        descs_db = res_descs_db
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
    qu_alphas = 0.5
    # List of datasets being used
    if largs.fit_db_tf_qu:
        use_ds = list(descs_db["database"])
    else:
        use_ds = list(descs_db)
    # Plot figure
    plt.figure()
    for db in use_ds:
        if largs.fit_db_tf_qu:
            plt.scatter(descs_db["database"][db][:, 0], 
                    descs_db["database"][db][:, 1],
                    label=db, c=db_colors[db], marker=db_markers[db])
            plt.scatter(descs_db["queries"][db][:, 0],
                    descs_db["queries"][db][:, 1], alpha=qu_alphas,
                    c=db_colors[db], marker=db_markers[db])
        else:
            plt.scatter(descs_db[db][:, 0], descs_db[db][:, 1], 
                    label=db, c=db_colors[db], marker=db_markers[db])
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
    
    res = load_descs(largs)
    # tSNE
    if use_caching and os.path.exists(f"{cache_fname}_tsne.gz"):
        print(f"Loading from cache file: {cache_fname}_tsne.gz")
        descs_db = joblib.load(f"{cache_fname}_tsne.gz")
    else:
        descs_db = tsne_project(largs, res)
        if use_caching:
            print(f"Saving to cache file: {cache_fname}_tsne.gz")
            joblib.dump(descs_db, f"{cache_fname}_tsne.gz")
    
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
