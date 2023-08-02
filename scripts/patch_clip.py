# Perform qualitative analysis of top-K retrievals
"""
    CLIP on patches of image -> VLAD
    
    DEPRECATED: This script is deprecated.
    
    Use CLIP as a global image descriptor and perform top-K retrievals
    from a given image sequence.
    The script assumes that the image sequence is of `datasets-vg`
    format.
    
    The path to the datasets folder has to be specified in configs (or
    argument).
    
    Definition of successful retrieval:
    A retrieval is successful if there is at least one database image
    retrieved in the top-k closest retrievals the to query.
    
    > Note: Currently, five crops and other methods are not 
        implemented. Only a single image `hard_resize` is implemented.
"""

# %%
# Python path gimmick
import os
import sys
from pathlib import Path
# Set the "./../" from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARNING: __file__ not found, trying local")
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f"{Path(dir_name).parent}")
# Add to path
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")


# %%
# Import everything
import numpy as np
import torch
import faiss
import fast_pytorch_kmeans as fpk
from torch.utils.data import DataLoader
from torchvision import transforms as tvf
import matplotlib.pyplot as plt
from PIL import Image
import traceback
from tqdm.auto import tqdm
import time
import joblib
# Internal packages
from dvgl_benchmark.datasets_ws import BaseDataset
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens

from clip_wrapper import ClipWrapper as Clip
from utilities import to_np, to_pil_list, pad_img, seed_everything
import pdb
import wandb


# %%
# Local configurations for this script
import tyro
from configs import device
from configs import ProgArgs, prog_args
from configs import BaseDatasetArgs, base_dataset_args
from dataclasses import dataclass, field
from typing import Literal, Union, Tuple, List

@dataclass
class LocalArgs:
    """
        Local arguments for the program
    """
    prog: ProgArgs = prog_args
    bd_args: BaseDatasetArgs = base_dataset_args
    # Batch size for processing images (set 1 for good cache)
    batch_size: int = 1
    # Experiment identifier for cache (set to False to disable cache)
    exp_id: Union[str, bool] = False
    """
        The results cache (joblib dump of 'dict') is saved in:
        - if `exp_id` is a string,
            caching_directory/experiments/`exp_id`
        - if `exp_id` is True, the images are stored in
            caching_directory
        - if `exp_id` is False, the result is not saved (only printed
            in the end)
        The file name is 'results_`timestamp`.gz'
    """
    # CLIP Implementation (OpenAI or Open CLIP)
    clip_impl: Literal["openai", "open_clip"] = Clip.IMPL_OPENAI
    # CLIP backbone architecture
    clip_backbone: str = "ViT-B/32"
    # CLIP pre-trained dataset (mainly for Open CLIP implementation)
    clip_pretrained: Union[None, str] = None
    # Dataset split (for dataloader)
    data_split: Literal["train", "test", "val"] = "test"
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Percentage of queries to save as qualitative results
    qual_result_percent: float = 0.025
    """
        If there are 1000 queries and this is 0.025, then there will 
        be about 25 images saved (sampled through uniform).
        Images are saved in the folder:
        - if `exp_id` is a string,
            caching_directory/experiments/`exp_id`/qualitative_retr
        - if `exp_id` is False, the images are not stored (since 
            caching is disabled)
        - if `exp_id` is True, the images are stored in
            caching_directory/qualitative_retr
        The images are saved as the file name (if saving is enabled): 
            `Q_{i_qu}_Top_{rec_i}.png` in the folder
        > Tip: If you set this to 0, then no images are saved
    """
    # Number of retrievals to show in the qualitative results
    qual_num_rets: int = 5
    # Similarity search
    faiss_method: Literal["l2", "cosine"] = "cosine"
    """
        Method (base index) to use for faiss nearest neighbor search.
        Find the complete table at [1].
        - "l2": The euclidean distances are used.
        - "cosine": The cosine distances (dot product) are used.
        
        Note that the descriptors given to the 'get_recalls' function
        are normalized beforehand.
        
        [1]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """
    # Use concatenation based Global Descriptors
    use_concat: Literal[1, 0] = bool(0)
    # Use summation based Global Descriptors
    use_sum: Literal[1, 0] = bool(0)
    # Use vlad based Global Descriptors
    use_vlad: Literal[1, 0] = bool(0)
    # Use soft assignment for VLAD encoding
    vlad_soft_assign: Literal[1, 0] = bool(0)
    # Softmax Temperature for soft-cluster assignment
    soft_assign_temp: int = 1
    # Number of Clusters for vlad based Global Descriptors
    num_clusters: int = 4
    # Number of Patches into which the image should be divided
    num_patches: int = 4

# Utility to visualize image
def visualize_tensor(tensor):
    minFrom= tensor.min()
    maxFrom= tensor.max()
    minTo = 0
    maxTo=1
    viz_tensor = minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))
    viz_tensor = viz_tensor.permute((1, 2, 0)).detach().cpu().numpy()
    return viz_tensor

# %%
# Build cache of all images in the dataset
def build_cache(largs: LocalArgs, vpr_dl: DataLoader, model: Clip) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Builds the cache files, and returns the descriptors for the
        database and queries (along with positives_per_query)
        
        Parameters:
        - largs:    Local arguments for this file
        - vpr_dl:   DataLoader for retrieving the images
        - model:    The CLIP model
        
        Returns:
        - db_descs:     Database image descriptors of shape 
                        [N_db, D=512]
        - qu_descs:     Query image descriptors of shape [N_qu, D=512]
        - pos_per_qu:   Positives (within a distance threshold) per
                        query index. [N_qu, ] list (object) with each
                        index containing positive sample indices.
    """
    full_res = []
    for batch in tqdm(vpr_dl):
        img_batch, ind_batch = batch[0], batch[1]
        assert img_batch.shape[0] == 1, "Batch size should be 1"
        # Get height and width of image
        img = img_batch[0]
        img_H, img_W = img.shape[1], img.shape[2]
        # Kernel and Stride size for creating patches
        kernel_size_h, stride_h = int(img_H * 2/largs.num_patches), int(img_H * 2/largs.num_patches)
        kernel_size_w, stride_w = int(img_W * 2/largs.num_patches), int(img_W * 2/largs.num_patches)
        # Convert Image to Patches
        patches = img.unfold(1, kernel_size_h, stride_h).unfold(2, kernel_size_w, stride_w)
        patches = patches.contiguous().view(patches.size(0), -1, kernel_size_h, kernel_size_w)
        patches = patches.permute((1, 0, 2, 3))
        # Encode Patches using CLIP
        with torch.no_grad():
            res = model.encode_image(patches, ci=int(ind_batch))
        # Append unsqueezed patch CLIP descriptors
        full_res.append(res.unsqueeze(0))
    # All Descriptors
    full_res = torch.concat(full_res, dim=0)
    full_res = full_res.detach().cpu()
    # Get Descriptors corresponding to Database and Query
    db_num = vpr_dl.dataset.database_num
    database_descs = full_res[:db_num]
    queries_descs = full_res[db_num:]
    
    return database_descs, queries_descs, vpr_dl.dataset.soft_positives_per_query

# %%
# Get recalls through similarity search
def get_recalls(largs: LocalArgs, ndb_descs: np.ndarray, 
            nqu_descs: np.ndarray, pos_per_qu: np.ndarray,
            vpr_dl:Union[None, DataLoader]=None, use_percentage=True, 
            use_gpu: bool=True, save_figs:bool= True):
    """
        Calculate the recalls through similarity search (using cosine
        distances).
        
        Parameters:
        - largs:    Local arguments to program. The following are used
                    - top_k_vals: For getting keys
        - ndb_descs:    Normalized database descriptors [N_d, D]
        - nqu_descs:    Normalized query descriptors [N_q, D]
        - pos_per_qu:   Positives (within a distance threshold) per
                        query index. [N_qu, ] list (object) with each
                        index containing positive sample indices.
        - vpr_dl:       DataLoader for images (used for getting 
                        qualitative results). Pass None if certain of
                        no qualitative results (see `save_figs`).
        - use_percentage:   If true, the recall is between [0, 1] and
                            not absolute. It's divided by N_q.
        - use_gpu:      Use GPU for faiss (else use CPU)
        - save_figs:    Save the qualitative results (if False, no
                        qualitative results are saved, and if True,
                        then saving depends on LocalArgs `exp_id`)
        
        Returns:
        - recalls: A dictionary of retrievals
    """
    # Saving preferences
    query_color = (125,   0, 125)   # RGB for query image (1st)
    false_color = (255,   0,   0)   # False retrievals
    true_color =  (  0, 255,   0)   # True retrievals
    padding = 20
    qimgs_result, qimgs_dir = True, \
        f"{largs.prog.cache_dir}/qualitative_retr" # Directory
    if largs.exp_id == False or largs.exp_id is None:   # Don't store
        qimgs_result, qimgs_dir = False, None
    elif type(largs.exp_id) == str:
        if not largs.use_vlad:
            qimgs_dir = f"{largs.prog.cache_dir}/experiments/"\
                        f"{largs.exp_id}/qualitative_retr"
        else:
            qimgs_dir = f"{largs.prog.cache_dir}/experiments/"\
                        f"{largs.exp_id}/qualitative_retr_vlad_nc{largs.num_clusters}"
    qimgs_inds = []
    if (not save_figs) or largs.qual_result_percent <= 0:
        qimgs_result = False
    if not qimgs_result:    # Saving query images
        print("Not saving qualitative results")
    else:
        _n_qu = nqu_descs.shape[0]
        qimgs_inds = np.random.default_rng().choice(
                range(_n_qu), int(_n_qu * largs.qual_result_percent),
                replace=False)  # Qualitative images to save
        print(f"There are {_n_qu} query images")
        print(f"Will save {len(qimgs_inds)} qualitative images")
        if not os.path.isdir(qimgs_dir):
            os.makedirs(qimgs_dir)  # Ensure folder exists
            print(f"Created qualitative directory: {qimgs_dir}")
        else:
            print(f"Saving qualitative results in: {qimgs_dir}")
    # FAISS search
    max_k = max(largs.top_k_vals)
    D = ndb_descs.shape[1]
    recalls = dict(zip(largs.top_k_vals, [0]*len(largs.top_k_vals)))
    if largs.faiss_method == "cosine":
        index = faiss.IndexFlatIP(D)
    elif largs.faiss_method == "l2":
        index = faiss.IndexFlatL2(D)
    else:
        raise Exception(f"FAISS method: {largs.faiss_method}!")
    if use_gpu:
        print("Running GPU faiss index")
        res = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(ndb_descs)    # Add database
    # Distances and indices are [N_q, max_k] shape
    distances, indices = index.search(nqu_descs, max_k) # Query
    for i_qu, qu_retr_maxk in enumerate(indices):
        for i_rec in largs.top_k_vals:
            correct_retr_qu = pos_per_qu[i_qu]  # Ground truth
            if np.any(np.isin(qu_retr_maxk[:i_rec], correct_retr_qu)):
                recalls[i_rec] += 1 # Query retrieved correctly
        if i_qu in qimgs_inds and qimgs_result:
            # Save qualitative results
            qual_top_k = qu_retr_maxk[:largs.qual_num_rets]
            correct_retr_qu = pos_per_qu[i_qu]
            color_mask = np.isin(qual_top_k, correct_retr_qu)
            colors_all = [true_color if x else false_color \
                        for x in color_mask]
            retr_dists = distances[i_qu, :largs.qual_num_rets]
            img_q = to_pil_list(    # Dataset is [database] + [query]
                vpr_dl.dataset[ndb_descs.shape[0]+i_qu][0])[0]
            img_q = to_np(img_q, np.uint8)
            # Main figure
            fig = plt.figure(figsize=(5*(1+largs.qual_num_rets), 5),
                            dpi=300)
            gs = fig.add_gridspec(1, 1+largs.qual_num_rets)
            ax = fig.add_subplot(gs[0, 0])
            ax.set_title(f"{i_qu} + {ndb_descs.shape[0]}")  # DS index
            ax.imshow(pad_img(img_q, padding, query_color))
            ax.axis('off')
            for i, db_retr in enumerate(qual_top_k):
                ax = fig.add_subplot(gs[0, i+1])
                img_r = to_pil_list(vpr_dl.dataset[db_retr][0])[0]
                img_r = to_np(img_r, np.uint8)
                ax.set_title(f"{db_retr} ({retr_dists[i]:.4f})")
                ax.imshow(pad_img(img_r, padding, colors_all[i]))
                ax.axis('off')
            fig.set_tight_layout(True)
            save_path = f"{qimgs_dir}/Q_{i_qu}_Top_{largs.qual_num_rets}.png"
            fig.savefig(save_path)
            plt.close(fig)
            if largs.prog.wandb_save_qual:
                wandb.log({"Qual_Results": wandb.Image(save_path)})
    if use_percentage:
        for k in recalls:
            recalls[k] /= len(indices)  # As a percentage of queries
    return recalls

# Function for VLAD Encoding
def get_vlad_vector(descriptors, cluster_centroids, cluster_labels, largs):
    """
    Computes the VLAD representation for a batch of local descriptors given a set of cluster centroids and cluster labels.

    Args:
        descriptors (torch.Tensor): A tensor of size (batch_size, num_patches, desc_dim), where batch_size is the batch size, num_patches is the number of local descriptors in each image and desc_dim is the dimensionality of each descriptor.
        cluster_centroids (torch.Tensor): A tensor of size (K, desc_dim), where K is the number of clusters and desc_dim is the dimensionality of each cluster centroid.
        cluster_labels (torch.Tensor): A tensor of size (batch_size, num_patches), where each element is an integer representing the index of the cluster centroid to which the corresponding descriptor is assigned.

    Returns:
        vlad_encoding (torch.Tensor): A tensor of size (batch_size, K * desc_dim), representing the VLAD encoding of the input descriptors for each image in the batch.
    """
    batch_size, num_patches, desc_dim = descriptors.shape
    num_clusters = cluster_centroids.shape[0]

    # Compute residuals for each descriptor
    residuals = descriptors.unsqueeze(2) - cluster_centroids.view(1, 1, -1, desc_dim)

    # Compute softmax-based cluster assignments
    soft_assign = torch.nn.functional.softmax(largs.soft_assign_temp * torch.nn.functional.cosine_similarity(descriptors.unsqueeze(2), cluster_centroids.view(1, 1, -1, desc_dim), dim=3), dim=2)

    # Compute VLAD encoding by summing residuals for each cluster centroid
    vlad_encoding = torch.zeros((batch_size, cluster_centroids.shape[0]*cluster_centroids.shape[1]), device=descriptors.device)
    for k in range(num_clusters):
        if largs.vlad_soft_assign:
            # Soft Cluster Assignment
            weight_mask = soft_assign[:, :, k].unsqueeze(-1).unsqueeze(-1)
        else:
            # Hard Cluster Assignment
            weight_mask = torch.zeros_like(residuals, device=residuals.device)
            weight_mask[cluster_labels == k, :, :] = 1
        residuals_k = residuals * weight_mask
        vlad_encoding_k = residuals_k.view(batch_size, -1, desc_dim).sum(dim=1)
        vlad_encoding_k = torch.nn.functional.normalize(vlad_encoding_k, dim=-1) # Intra-Normalization
        vlad_encoding[:, k*desc_dim:(k+1)*desc_dim] = vlad_encoding_k

    # L2-normalize the VLAD encoding
    vlad_encoding = torch.nn.functional.normalize(vlad_encoding, dim=-1)
    
    return vlad_encoding

# %%
# Main function
def main():
    largs = tyro.cli(LocalArgs)
    print(f"Arguments: {largs}")
    seed_everything()

    # Launch Wandb
    wandb.init(project=largs.prog.wandb_proj, entity=largs.prog.wandb_entity, config=largs, group=largs.prog.wandb_group, name=largs.prog.wandb_run_name)
    
    print("------------------ CLIP Model ------------------")
    model = Clip(largs.clip_impl, largs.clip_backbone, 
                largs.clip_pretrained, use_caching=largs.exp_id, 
                base_cache_dir=largs.prog.cache_dir,
                device=device)
    print("-------------- CLIP model loaded --------------")
    
    print("-------- Generating Patch Descriptors --------")
    datasets_dir = largs.prog.data_vg_dir
    dataset_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {datasets_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset split: {largs.data_split}")

    if dataset_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)

    elif dataset_name=="Oxford":
        vpr_ds = Oxford(datasets_dir,use_mixVPR=True)

    elif dataset_name=="gardens":
        vpr_ds = Gardens(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)
    else:
        vpr_ds = BaseDataset(largs.bd_args, datasets_dir, dataset_name, 
                            largs.data_split,use_mixVPR=True)


    vpr_dl = DataLoader(vpr_ds, largs.batch_size, pin_memory=True, shuffle=False)
    db_descs, qu_descs, pos_pq = build_cache(largs, vpr_dl, model)
    # Normalize the descriptors
    ndb_descs = torch.nn.functional.normalize(db_descs, p=2, dim=-1)
    nqu_descs = torch.nn.functional.normalize(qu_descs, p=2, dim=-1)
    print("-------- Patch descriptors generated --------")

    if largs.use_concat:
        print("-------- Generating Global Descriptors based on Concatenation --------")
        # Concatenate Patch Descriptors
        ndb_descs = torch.flatten(ndb_descs, start_dim=1)
        nqu_descs = torch.flatten(nqu_descs, start_dim=1)
        # L2 Normalize Final Concat Descriptors
        ndb_descs = torch.nn.functional.normalize(ndb_descs, p=2, dim=-1)
        nqu_descs = torch.nn.functional.normalize(nqu_descs, p=2, dim=-1)
        print("-------- Concat Global Descriptors generated --------")
    elif largs.use_vlad:
        print("-------- Generating VLAD based Global Descriptors --------")
        print("-------- Num  of Clusters: {} --------".format(largs.num_clusters))
        # Patch Descriptors of all Database images
        ndb_patch_descs = ndb_descs.view(-1, ndb_descs.shape[2])
        # Kmeans on Database Patch Descriptors
        kmeans = fpk.KMeans(n_clusters=largs.num_clusters, mode='cosine')
        labels = kmeans.fit_predict(ndb_patch_descs)
        db_cluster_labels = labels.view(ndb_descs.shape[0], -1)
        # Cluser Labels of Query Patch Descriptors
        nqu_patch_descs = nqu_descs.view(-1, nqu_descs.shape[2])
        qu_labels = kmeans.predict(nqu_patch_descs)
        qu_cluster_labels = qu_labels.view(nqu_descs.shape[0], -1)
        # VLAD Global Descriptors based on Patch Descriptors
        ndb_descs = get_vlad_vector(ndb_descs, kmeans.centroids, db_cluster_labels, largs)
        nqu_descs = get_vlad_vector(nqu_descs, kmeans.centroids, qu_cluster_labels, largs)
        print("-------- VLAD based Global descriptors generated --------")
    elif largs.use_sum:
        print("-------- Generating Global Descriptors based on Summation --------")
        # Sum Patch Descriptors
        ndb_descs = torch.sum(ndb_descs, dim=1)
        nqu_descs = torch.sum(nqu_descs, dim=1)
        # L2 Normalize Final Summation Descriptors
        ndb_descs = torch.nn.functional.normalize(ndb_descs, p=2, dim=-1)
        nqu_descs = torch.nn.functional.normalize(nqu_descs, p=2, dim=-1)
        print("-------- Summation Global Descriptors generated --------")
    else:
        raise NotImplementedError
    
    # Convert Torch tensors to Numpy arrays
    ndb_descs = ndb_descs.numpy()
    nqu_descs = nqu_descs.numpy()

    print("----------- FAISS Search started -----------")
    recalls = get_recalls(largs, ndb_descs, nqu_descs, pos_pq, vpr_dl)
    print("------------ FAISS Search ended ------------")
    
    print("----------------- Results -----------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "CLIP-impl": str(largs.clip_impl),
        "CLIP-backbone": str(largs.clip_backbone),
        "CLIP-pretrained": str(largs.clip_pretrained),
        "Experiment-ID": str(largs.exp_id),
        "Cache-dir": str(largs.prog.cache_dir),
        "Dataset-name": str(dataset_name),
        "Timestamp": str(ts),
        "FAISS-metric": largs.faiss_method,
        "Num of Patches": largs.num_patches,
        "Use-Concat-Agg": largs.use_concat,
        "Use-VLAD-Agg": largs.use_vlad,
        "Num of Clusters": largs.num_clusters
    }
    for tk in recalls.keys():
        key = f"R@{tk}"
        results[key] = recalls[tk]
    print("Results:")
    for k in results:
        print(f"- {k}: {results[k]}")

    # Log to Wandb
    wandb.log(results)
    # Close Wandb
    wandb.finish()

    save_res_file = None
    if largs.exp_id == True:
        save_res_file = caching_directory
    elif type(largs.exp_id) == str:
        save_res_file = f"{caching_directory}/experiments/"\
                        f"{largs.exp_id}"
    if save_res_file is not None:
        if not os.path.isdir(save_res_file):
            os.makedirs(save_res_file)
        if not largs.use_vlad:
            save_res_file = f"{save_res_file}/results_{ts}.gz"
        else:
            save_res_file = f"{save_res_file}/results_vlad_nc{largs.num_clusters}_{ts}.gz"
        print(f"Saving result in: {save_res_file}")
        joblib.dump(results, save_res_file)
    else:
        print("Not saving results")
    print("--------------------- END ---------------------")


# Main entrypoint
if __name__ == "__main__" and (not "ipykernel" in sys.argv[0]):
    try:
        main()
    except (Exception, SystemExit) as exc:
        print(f"Exception: {exc}")
        if str(exc) == "0":
            print("[INFO]: Exit is safe")
        else:
            print("[ERROR]: Exit is not safe")
            traceback.print_exc()


# %%
