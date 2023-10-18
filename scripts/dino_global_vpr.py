# Perform VPR on Dino descriptors
"""
    Use the CLS token (facet = "token") of the last layer to be the 
    global descriptor for the image. Use it for VPR.
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
from dino_extractor import ViTExtractor
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
from custom_datasets.hawkins_dataloader import Hawkins
from custom_datasets.vpair_dataloader import VPAir
from custom_datasets.vpair_distractor_dataloader import VPAir_Distractor
from custom_datasets.laurel_dataloader import Laurel
from custom_datasets.eiffel_dataloader import Eiffel


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
    # Dino parameters
    model_type: Literal["dino_vits8", "dino_vits16", "dino_vitb8", 
            "dino_vitb16", "vit_small_patch8_224", 
            "vit_small_patch16_224", "vit_base_patch8_224", 
            "vit_base_patch16_224"] = "dino_vits8"
    """
        Model for Dino to use as the base model.
    """
    # Stride for ViT (extractor)
    vit_stride: int = 4
    # Down-scaling H, W resolution for images (before giving to Dino)
    down_scale_res: Tuple[int, int] = (224, 298)
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Apply log binning to the descriptor
    desc_bin: bool = False
    # Sub-sample query images (RAM or VRAM constraints) (1 = off)
    sub_sample_qu: int = 1
    # Sub-sample database images (RAM or VRAM constraints) (1 = off)
    sub_sample_db: int = 1
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Similarity search
    faiss_method: Literal["l2", "cosine"] = "cosine"
    """
        Method (base index) to use for faiss nearest neighbor search.
        Find the complete table at [1].
        - "l2": The euclidean distances are used.
        - "cosine": The cosine distances (dot product) are used.
        
        Note that `get_top_k_recall` normalizes the descriptors given 
        as input.
        
        [1]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """
    # Show a matplotlib plot for recalls
    show_plot: bool = False


# %%
@torch.no_grad()
def build_cache(largs: LocalArgs, vpr_ds: BaseDataset, 
            verbose: bool=True, vpr_distractor: BaseDataset=None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Build global descriptors for database and query images.
        
        Parameters:
        - largs: LocalArgs  Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - verbose: bool     Prints progress if True
        
        Returns:
        - db_descs: Database image descriptors of shape [N_db, m_d]
                        - N_db: Number of database images
                        - m_d: Dimensionality of descriptor (last 
                            layer, facet token, CLS). Depends on the
                            model.
        - qu_descs: Query image descriptors of shape [N_qu, m_d]
    """
    extractor = ViTExtractor(largs.model_type, largs.vit_stride, 
                device=device)
    
    def extract_gd(indices, use_distractor: bool=False):
        full_descs = []
        for i in tqdm(indices, disable=not verbose):
            if use_distractor:
                img = vpr_distractor[i][0]
            else:
                img = vpr_ds[i][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            desc = extractor.extract_descriptors(img,
                layer=11, facet="token", bin=largs.desc_bin, 
                include_cls=True) # [1, 1, 1 + n_d, d_dim]
            full_descs.append(desc[0, 0, [0]])  # n_d = 55 * 73
        full_descs = torch.cat(full_descs, dim=0)
        return full_descs
    
    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    full_db_descs = extract_gd(db_indices)
    if verbose:
        print(f"Database descriptors shape: {full_db_descs.shape}")
    # Get query descriptors
    qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    full_qu_descs = extract_gd(qu_indices)
    if verbose:
        print(f"Query descriptors shape: {full_qu_descs.shape}")
    # Get database distractor descriptors
    if vpr_distractor is not None:
        num_db_distractor = vpr_distractor.database_num
        if verbose:
            print("Extracting global descriptors for vpair distractors...")
        try:
            db_dis_indices = np.arange(0, num_db_distractor, largs.sub_sample_db)
            full_db_descs_dis = extract_gd(db_dis_indices, use_distractor=True)
            full_db_descs = torch.cat((full_db_descs, full_db_descs_dis), dim=0)
            if verbose:
                print(f"Database with distractors shape: {full_db_descs.shape}")
        except RuntimeError as exc:
            print(f"Runtime error: {exc}")
            print("Not using vpair distractors")
    
    return full_db_descs, full_qu_descs


# %%
@torch.no_grad()
def main(largs: LocalArgs):
    """
        Main function
    """
    print(f"Arguments: {largs}")
    seed_everything()
    
    if largs.prog.use_wandb:
        # Launch WandB
        wandb_run = wandb.init(project=largs.prog.wandb_proj,
                entity=largs.prog.wandb_entity, config=largs,
                group=largs.prog.wandb_group, 
                name=largs.prog.wandb_run_name)
        print(f"Initialized WandB run: {wandb_run.name}")
    
    print("------------ Loading dataset ------------")
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}")
    print(f"Dataset split: {largs.data_split}")
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
    print("------------ Dataset loaded ------------")
    
    print("------- Generating global descriptors -------")
    if ds_name == "VPAir":
        db_descs, qu_descs = build_cache(largs, vpr_ds, vpr_distractor=vpr_distractor_ds)
    else:
        db_descs, qu_descs = build_cache(largs, vpr_ds)
    print("------- Global descriptors generated -------")
    
    print("----------- FAISS Search started -----------")
    # u = True if device.type == "cuda" else False
    u = False   # TODO: debugging this
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
            db_descs.cpu(), qu_descs.cpu(), vpr_ds.get_positives(), 
            method=largs.faiss_method, use_gpu=u,
            sub_sample_db=largs.sub_sample_db, 
            sub_sample_qu=largs.sub_sample_qu)
    print("------------ FAISS Search ended ------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Model-Type": str(largs.model_type),
        "DB-Name": str(largs.prog.vg_dataset_name),
        "Timestamp": str(ts),
        "FAISS-metric": str(largs.faiss_method),
        "Agg-Method": "Global",
        "Desc-Dim": str(db_descs.shape[1])
    }
    print("Results:")
    for k in results:
        print(f"- {k}: {results[k]}")
    print("- Recalls:")
    for tk in recalls.keys():
        results[f"R@{tk}"] = recalls[tk]
        print(f"  - R@{tk}: {recalls[tk]:.5f}")
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
    print("--------------------- END ---------------------")


# %%
if __name__ == "__main__" and (not "ipykernel" in sys.argv[0]):
    largs = tyro.cli(LocalArgs)
    _start = time.time()
    try:
        main(largs)
    except (Exception, SystemExit) as exc:
        print(f"Exception: {exc}")
        if str(exc) == "0":
            print("[INFO]: Exit is safe")
        else:
            print("[ERROR]: Exit is not safe")
            traceback.print_exc()
    except:
        print("Unhandled error")
        traceback.print_exc()
    finally:
        print(f"Program ended in {time.time()-_start:.3f} seconds")
        exit(0)


# %%
