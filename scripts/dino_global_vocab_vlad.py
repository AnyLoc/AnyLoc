# Get VLAD cluster centers by a global collection of images
"""
    DINOv1 vocabulary generation using a global collection of images.
    
    See the description of `dino_v2_global_vocab_vlad.py`
    
    TL;DR: Use `--db-samples` to selective (sub)sample datasets for
        tailoring the VLAD cluster centers (vocabulary).
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
import einops as ein
from PIL import Image
from dino_extractor import ViTExtractor
from torchvision import transforms as tvf
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, Literal
from tqdm.auto import tqdm
import traceback
import wandb
import time
import joblib
import tyro
import matplotlib.pyplot as plt
from utilities import VLAD, get_top_k_recall, seed_everything
from utilities import DinoV2ExtractFeatures
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
    # Program arguments (dataset directories and wandb only)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-v1-Descs", 
        wandb_group="VLAD-Descs")
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use) [won't be used for caching]
    exp_id: Union[str, None] = None
    # VLAD Caching directory (None = don't cache)
    vlad_cache_dir: Path = None
    # VLAD Caching for the database and query
    vlad_cache_db_qu: bool = False
    """
        If the `vlad_cache_dir` is not None (then VLAD caching is 
        turned on), this flag controls whether the database and query
        images are cached. If False, then only the cluster centers are
        cached. If True, then database and query image VLADs are also
        cached. This is controlled by the ID for caching (it's made
        None in case of no caching).
    """
    # Resize the image (doesn't work, always 320, 320)
    resize: Tuple[int, int] = field(default_factory=lambda: (320, 320))
    # Number of clusters for VLAD
    num_clusters: int = 8
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Dino parameters
    # Model type
    model_type: Literal["dino_vits8", "dino_vits16", "dino_vitb8", 
            "dino_vitb16", "vit_small_patch8_224", 
            "vit_small_patch16_224", "vit_base_patch8_224", 
            "vit_base_patch16_224"] = "dino_vits8"
    """
        Model for Dino-v2 to use as the base model.
    """
    # Down-scaling H, W resolution for images (before giving to Dino)
    down_scale_res: Tuple[int, int] = (224, 298)
    # Layer for extracting Dino feature (descriptors)
    desc_layer: int = 11
    # Facet for extracting descriptors
    desc_facet: Literal["query", "key", "value", "token"] = "key"
    # Apply log binning to the descriptor
    desc_bin: bool = False
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
    # Stride for ViT (extractor)
    vit_stride: int = 4
    # Down-scaling H, W resolution for images (before giving to Dino)
    down_scale_res: Tuple[int, int] = (224, 298)
    # Databases to sample
    db_samples: dict = field(default_factory=lambda: {  # Database name: sub-sampling frequency
        "Oxford": 0,
        "gardens": 0,
        "17places": 0,
        "baidu_datasets": 0,
        "st_lucia": 0,
        "pitts30k": 0,
        "Tartan_GNSS_test_rotated": 0,
        "Tartan_GNSS_test_notrotated": 0,
        "hawkins": 0,
        "laurel_caverns": 0,
        "eiffel": 0,
        "VPAir": 0
    })
    """
        Configure the sampling of database images for different 
        datasets. The key is the dataset name and the value is the
        sub-sampling frequency (applied to database images) - note
        that this sub-sampling is before the `sub_sample_db_vlad`.
        To remove a dataset from here, set its value to 0 (it'll be
        dropped when loading VLAD).
    """
    # Save Database and Query VLAD descriptors (final)
    save_vlad_descs: Optional[Path] = None
    """
        Internal use only (don't set normally). Save the database and
        query VLAD descriptors to this folder. The file name is
        `db-<prog.vg_dataset_name>.pt` (for database VLADs) and 
        `qu-<prog.vg_dataset_name>.pt` (for query VLADs).
        This is independent of the caching mechanisms.
    """


# %%
# Global dataset class
class GlobalVLADVocabularyDataset:
    """
        A global wrapper class to create a concatenated list of all
        passed database images. Also has options for sub-sampling.
    """
    def __init__(self, ds_names: List[str], ds_dir:str, ds_split:str,
            bd_args: BaseDatasetArgs=base_dataset_args,
            ss_list: Union[int, List[int]]=1, 
            size: Tuple[int, int]=(320, 320)):
        """
            Parameters:
            - ds_names:     A list of dataset names (IDs) to use
            - ds_dir:       Directory where the datasets are stored
            - ds_split:     Split to use (some datasets need this)
            - bd_args:      Base dataset arguments to use (shared)
            - ss_list:      A list of sub-sampling values to use (for
                            database images). If int, then all 
                            sub-sampling values are the same for all 
                            datasets.
            - size:         Image size to resize to (when reading)
        """
        self.ds_names = ds_names
        if type(ss_list) == int:
            self.ss_list = [ss_list] * len(ds_names)
        else:
            self.ss_list = ss_list
        self.base_transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            tvf.Resize(size)
        ])
        # Load all dataset images
        print(f"Dataset directory: {ds_dir}")
        print(f"Dataset split: {ds_split}")
        data_split = ds_split
        self.db_img_paths = []
        self.db_stat = {}
        for i, ds_name in enumerate(ds_names):
            print(f"Dataset: {ds_name} ->", end=" ")
            if ds_name=="baidu_datasets":
                vpr_ds = Baidu_Dataset(bd_args, ds_dir, ds_name, 
                        data_split)
            elif ds_name=="Oxford":
                vpr_ds = Oxford(ds_dir)
            elif ds_name=="Oxford_25m": # This is actually useless!
                vpr_ds = Oxford(ds_dir, override_dist=25)
            elif ds_name=="gardens":
                vpr_ds = Gardens(bd_args, ds_dir, ds_name, data_split)
            elif ds_name.startswith("Tartan_GNSS"):
                vpr_ds = Aerial(bd_args, ds_dir, ds_name, data_split)
            elif ds_name.startswith("hawkins"): # Use long_corridor
                vpr_ds = Hawkins(bd_args, ds_dir,
                        "hawkins_long_corridor", data_split)
            elif ds_name=="VPAir":
                vpr_ds = VPAir(bd_args, ds_dir, ds_name, data_split)
                vpr_distractor_ds = VPAir_Distractor(bd_args,
                        ds_dir, ds_name, data_split)
            elif ds_name=="laurel_caverns":
                vpr_ds = Laurel(bd_args, ds_dir, ds_name, data_split)
            elif ds_name=="eiffel":
                vpr_ds = Eiffel(bd_args, ds_dir, ds_name, data_split)
            else:
                vpr_ds = BaseDataset(bd_args, ds_dir, ds_name, 
                        data_split)
            imgs_path = vpr_ds.get_image_paths()
            num_db = vpr_ds.database_num
            num_ss = self.ss_list[i]
            print(f"{len(imgs_path)} images {num_db} DB ->", end=" ")
            db_imgs_path = imgs_path[:num_db:num_ss]
            self.db_img_paths.extend(db_imgs_path)
            self.db_stat[ds_name] = len(db_imgs_path)
            print(f"{len(db_imgs_path)} (used DB)")
        self.database_num = len(self.db_img_paths)
        print(f"All database images: {self.database_num}")
        self.images_paths = self.db_img_paths
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self) -> str:
        return f"Composition: {self.db_stat}"
    
    def __getitem__(self, idx):
        img = Image.open(self.images_paths[idx])
        img = self.base_transform(img)
        return img, idx


# %%
# %%
@torch.no_grad()
def build_vlads_fm_global(largs: LocalArgs, vpr_ds: BaseDataset, 
        glob_ds: GlobalVLADVocabularyDataset, verbose: bool=True,
        vpr_distractor_ds: BaseDataset=None):
    """
        Build VLAD vectors for database and query images using the
        cluster centers from the global vocabulary (collection) data.
        
        Parameters:
        - largs: LocalArgs  Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - glob_ds:      The global vocabulary dataset (for cluster
                        centers only).
        - verbose: bool     Prints progress if True
        - vpr_distractor_ds: BaseDataset
                A dataset containing distractor images (if applicable)
                that are concatenated with the 
    """
    cache_dir = largs.vlad_cache_dir
    if cache_dir is not None:
        print(f"Using cache directory: {cache_dir}")
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
            print(f"Directory created: {cache_dir}")
        else:
            print(f"Directory already exists: {cache_dir}")
    
    vlad = VLAD(largs.num_clusters, None, 
            vlad_mode=largs.vlad_assignment, 
            soft_temp=largs.vlad_soft_temp, cache_dir=cache_dir)
    # Load Dino feature extractor model
    extractor = ViTExtractor(largs.model_type, largs.vit_stride,
                device=device)
    if verbose:
        print("Dino model loaded")
    
    def extract_patch_descriptors(indices, 
            use_set: Literal["vpr", "distractor", "global"]="vpr"):
        patch_descs = []
        for i in tqdm(indices, disable=not verbose):
            if use_set == "vpr":
                img = vpr_ds[i][0]
            elif use_set == "distractor":
                img = vpr_distractor_ds[i][0]
            elif use_set == "global":
                img = glob_ds[i][0]
            else:
                raise ValueError(f"Invalid use set: {use_set}")
            # c, h, w = img.shape
            # h_new, w_new = (h // 14) * 14, (w // 14) * 14
            # img_in = tvf.CenterCrop((h_new, w_new))(img)[None, ...]
            # ret = dino(img_in.to(device))
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            ret = extractor.extract_descriptors(img,
                    layer=largs.desc_layer, facet=largs.desc_facet,
                    bin=largs.desc_bin) # [1, 1, num_descs, d_dim]
            patch_descs.append(ret.cpu()[0])
        patch_descs = torch.cat(patch_descs, dim=0) # [N, n_p, d_dim]
        return patch_descs
    
    # Check for cluster centers
    if vlad.can_use_cache_vlad():
        if verbose:
            print("Using cached VLAD cluster centers")
        vlad.fit(None)
    else:
        # Get cluster centers using global voccabulary
        if verbose:
            print("Building VLAD cluster centers...")
        num_db = len(glob_ds)
        db_indices = np.arange(0, num_db, largs.sub_sample_db_vlad)
        # Database descriptors (for VLAD clusters): [n_db, n_d, d_dim]
        full_db_vlad = extract_patch_descriptors(db_indices, "global")
        if verbose:
            print(f"Database descriptors shape: {full_db_vlad.shape}")
        d_dim = full_db_vlad.shape[2]
        if verbose:
            print(f"Descriptor dimensionality: {d_dim}")
        vlad.fit(ein.rearrange(full_db_vlad, "n k d -> (n k) d"))
        del full_db_vlad
    if verbose:
        print(f"VLAD cluster centers shape: "\
                f"{vlad.c_centers.shape}, ({vlad.c_centers.dtype})")
    
    # Database images
    c_dbq = largs.vlad_cache_db_qu
    if verbose:
        print("Building VLADs for database...")
    num_db = vpr_ds.database_num
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    db_img_names = vpr_ds.get_image_relpaths(db_indices)
    if c_dbq and vlad.can_use_cache_ids(db_img_names):
        if verbose:
            print("Using cached VLADs for database images")
        db_vlads = vlad.generate_multi([None] * len(db_img_names), 
                db_img_names)
    else:
        if verbose:
            print("Valid cache not found, doing forward pass")
        full_db = extract_patch_descriptors(db_indices, "vpr")
        if not c_dbq:
            db_img_names = [None] * len(db_img_names)
        db_vlads: torch.Tensor = vlad.generate_multi(full_db, 
                db_img_names)
        del full_db
    if verbose:
        print(f"Database VLADs shape: {db_vlads.shape}")
    
    # Get VLADs of the queries
    if verbose:
        print("Building VLADs for query images...")
    ds_len = len(vpr_ds)
    q_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    qu_img_names = vpr_ds.get_image_relpaths(q_indices)
    if c_dbq and vlad.can_use_cache_ids(qu_img_names):
        if verbose:
            print("Using cached VLADs for query images")
        qu_vlads = vlad.generate_multi([None] * len(qu_img_names), 
                qu_img_names)
    else:
        if verbose:
            print("Valid cache not found, doing forward pass")
        full_qu = extract_patch_descriptors(q_indices, "vpr")
        if not c_dbq:
            qu_img_names = [None] * len(qu_img_names)
        qu_vlads = vlad.generate_multi(full_qu,
                qu_img_names)
        del full_qu
    if verbose:
        print(f"Query VLADs shape: {qu_vlads.shape}")
    
    # Append to db_vlads for vpair distractors
    if vpr_distractor_ds is not None:
        num_dis_db = vpr_distractor_ds.database_num
        if verbose:
            print("Building VLADs for vpair distractors...")
        try:
            db_dis_indices = np.arange(0, num_dis_db, 
                    largs.sub_sample_db)
            db_dis_img_names = vpr_distractor_ds.get_image_relpaths(
                    db_dis_indices)
            if c_dbq and vlad.can_use_cache_ids(db_dis_img_names):
                if verbose:
                    print("Valid cache found, using it")
                db_dis_vlads = vlad.generate_multi([None] * len(
                        db_dis_img_names), db_dis_img_names)
            else:
                if verbose:
                    print("Valid cache not found, doing forward pass")
                full_dis_db = extract_patch_descriptors(
                        db_dis_indices, "distractor")
                if verbose:
                    print(f"Dist. VLAD shape: {full_dis_db.shape}")
                if not c_dbq:
                    db_dis_img_names = [None] * len(db_dis_img_names)
                db_dis_vlads: torch.Tensor = vlad.generate_multi(
                        full_dis_db, db_dis_img_names)
                del full_dis_db
            if verbose:
                print(f"Dist. VLAD shape: {db_dis_vlads.shape}")
            c_db_vlads = torch.concatenate((db_vlads,db_dis_vlads),0)
            db_vlads = c_db_vlads
            if verbose:
                print(f"Combined db VLAD shape: {db_vlads.shape}")
        except RuntimeError as exc:
            print(f"Runtime error: {exc}")
            traceback.print_exc()
            print("Ignoring vpair distractors")
    
    return db_vlads, qu_vlads


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
    
    print("--------- Loading datasets ---------")
    ds_dir = largs.prog.data_vg_dir
    ds_split = largs.data_split
    print(f"Dataset directory: {ds_dir}")
    ds_use = [ds for ds in largs.db_samples \
            if largs.db_samples[ds] != 0]
    assert len(ds_use) > 0, "No datasets selected"
    glob_ds = GlobalVLADVocabularyDataset(ds_use, ds_dir, ds_split, 
            largs.bd_args, [largs.db_samples[k] for k in ds_use])
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset name (to use): {ds_name}")
    # Load dataset
    if ds_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(largs.bd_args, ds_dir, ds_name, 
                            largs.data_split)
    elif ds_name=="Oxford":
        vpr_ds = Oxford(ds_dir)
    elif ds_name=="Oxford_25m":
        vpr_ds = Oxford(ds_dir, override_dist=25)
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
    
    if ds_name=="VPAir":
        db_vlads, qu_vlads = build_vlads_fm_global(largs, vpr_ds,
                glob_ds, vpr_distractor_ds=vpr_distractor_ds)
    else:
        db_vlads, qu_vlads = build_vlads_fm_global(largs, vpr_ds, 
                glob_ds)
    print("--------- Generated VLADs ---------")
    
    print("----- Calculating recalls through top-k matching -----")
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
        db_vlads, qu_vlads, vpr_ds.soft_positives_per_query, 
        sub_sample_db=largs.sub_sample_db, 
        sub_sample_qu=largs.sub_sample_qu)
    print("------------ Recalls calculated ------------")
    
    # If saving (for internal debugging only)
    if largs.save_vlad_descs is not None:
        print("------ Saving VLAD descriptors ------")
        print(f"DB VLAD shape: {db_vlads.shape}")
        print(f"QU VLAD shape: {qu_vlads.shape}")
        save_dir = os.path.realpath(os.path.expanduser(
                largs.save_vlad_descs))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        else:
            print(f"Save directory already exists: {save_dir}")
        # Save files
        torch.save(db_vlads.cpu(), f"{save_dir}/db-{ds_name}.pt")
        torch.save(qu_vlads.cpu(), f"{save_dir}/qu-{ds_name}.pt")
        print(f"Saved files [db,qu]-{ds_name}.pt in {save_dir}")
    
    print("--------------------- Results ---------------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Model-Type": str(largs.model_type),
        "Desc-Layer": str(largs.desc_layer),
        "Desc-Facet": str(largs.desc_facet),
        "Desc-Dim": str(db_vlads.shape[1]//largs.num_clusters),
        "VLAD-Dim": str(db_vlads.shape[1]),
        "Num-Clusters": str(largs.num_clusters),
        "Experiment-ID": str(largs.exp_id),
        "DB-Name": str(ds_name),
        "Num-DB": str(len(db_vlads)),
        "Num-QU": str(len(qu_vlads)),
        "Agg-Method": "VLAD",
        "Timestamp": str(ts)
    }
    print("Results: ")
    for k in results:
        print(f"- {k}: {results[k]}")
    print("- Recalls: ")
    for k in recalls:
        results[f"R@{k}"] = recalls[k]
        print(f"  - R@{k}: {recalls[k]:.5f}")
    if largs.show_plot:
        plt.plot(recalls.keys(), recalls.values())
        plt.ylim(0, 1)
        plt.xticks(largs.top_k_vals)
        plt.xlabel("top-k values")
        plt.ylabel(r"% recall")
        plt_title = "Recall curve"
        if largs.exp_id is not None:
            plt_title = f"{plt_title} - Exp {largs.exp_id}"
        if largs.prog.use_wandb:
            plt_title = f"{plt_title} - {wandb_run.name}"
        plt.title(plt_title)
        plt.show()
    
    # Log to WandB
    if largs.prog.use_wandb:
        wandb.log(results)
        for tk in recalls:
            wandb.log({"Recall-All": recalls[tk]}, step=int(tk))
    
    # Add retrievals
    results["Qual-Dists"] = dists
    results["Qual-Indices"] = indices
    save_res_file = None
    if largs.exp_id == True:
        save_res_file = caching_directory
    elif type(largs.exp_id) == str:
        save_res_file = f"{caching_directory}/experiments/"\
                        f"{largs.exp_id}"
    if save_res_file is not None:
        if not os.path.isdir(save_res_file):
            os.makedirs(save_res_file)
        save_res_file = f"{save_res_file}/results_{ts}.gz"
        print(f"Saving result in: {save_res_file}")
        joblib.dump(results, save_res_file)
    else:
        print("Not saving results")
    
    if largs.prog.use_wandb:
        wandb.finish()
    print("--------------------- END ---------------------")


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
