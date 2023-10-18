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


import torch
from torch.optim import lr_scheduler, optimizer
from models import helper
import cv2

##SAM and DINO imports
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dino_extractor import ViTExtractor

class VPRModel(torch.nn.Module):
    """
    This is the modified main model for Visual Place Recognition without Pytorch lightning
    """
    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
        # if using SAM
        if self.encoder_arch=='SAM':
            self.sam = sam_model_registry["vit_l"](checkpoint="/ocean/projects/cis220039p/jkarhade/data/sam_model/sam_vit_l_0b3195.pth")
            self.sam.to(device)
            self.predictor = SamPredictor(self.sam)

        # if using DINO
        if self.encoder_arch=='DINO':
            self.extractor = ViTExtractor('dino_vits8', 4,device=device)

    # the forward pass of the lightning model
    def forward(self, x):
        # print("original",x.shape)
        ##If using SAM
        if self.encoder_arch=='SAM':
            x = x.squeeze(0).detach().cpu().numpy()
            self.predictor.set_image(x)
            x = self.predictor.get_image_embedding()
        elif self.encoder_arch=='DINO':
            img = x.to(device)
            img = F.interpolate(img, (224,298))
            # print(img.shape)
            x = self.extractor.extract_descriptors(img,
                    layer=6, facet='key',
                    bin=False) # [1, 1, num_descs, d_dim]
            x = x.reshape(1,55,73,-1).permute(0,3,1,2)
        else:
            x = self.backbone(x)
        # print("backbone",x.shape)
        x = self.aggregator(x)
        # print("aggregator",x.shape)
        return x

# Perform qualitative analysis of top-K retrievals
"""
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
    
    Results dictionary keys:
    - CLIP-impl: CLIP implementation (OpenAI or Open)
    - CLIP-backbone: Backbone for CLIP
    - Experiment-ID: Unique experiment ID (LocalArg: '--exp-id')
    - Cache-dir: Cache directory for the experiment
    - Dataset-name: Dataset name (on which VPR was done)
    - Timestamp: Timestamp in "Year_Month_Day_Hour_Min_Sec"
    - PCA-dim: PCA dimensionality (for reduction)
    - PCA-lower-factor: The fraction of lower eigen-basis vectors
    - FAISS-metric: Distance metric for nearest-neighbor
    - VLAD-nclusters: Number of cluster centers for VLAD-style 
            aggregation of descriptor residuals
    - Recall: Recall dictionary (keys are 'k' - as in top-k)
    - PCA-dim: PCA dimensionality (for reduction)
    - PCA-lower-factor: The fraction of lower eigen-basis vectors
    - FAISS-metric: Distance metric for nearest-neighbor
    - Recall: Recall values for each top-k thresholds
"""

# %%
# Import everything
import numpy as np
import torch
from torch.nn import functional as F
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
from custom_datasets.aerial_dataloader import Aerial
from custom_datasets.hawkins_dataloader import Hawkins
from custom_datasets.vpair_dataloader import VPAir
from custom_datasets.laurel_dataloader import Laurel
from custom_datasets.eiffel_dataloader import Eiffel
from custom_datasets.vpair_distractor_dataloader import VPAir_Distractor
from clip_wrapper import ClipWrapper as Clip
from utilities import to_np, to_pil_list, pad_img, reduce_pca, \
    concat_desc_dists_clusters, seed_everything
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
    # Checkpoint location
    ckpt_file: Path = '/ocean/projects/cis220039p/jkarhade/vl-vpr/MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4) (1).ckpt'
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
    # PCA dimensionality reduction
    pca_dim_reduce: Union[int, None] = None
    """
        If 'None', no PCA is applied to the database and query images.
        Otherwise, the dimension of database and query image 
        descriptors are reduced to this integer.
    """
    # Take eigen-basis corresponding the lowest eigenvalues
    pca_low_factor: float = 0.0
    """
        This is the fraction of basis vectors that are taken from the
        lower eigenvalues. If 0.0, all 100% of pca_dim_reduce are
        taken from largest eigenvalues (default PCA). If 1.0, only the
        lowest pca_dim_reduce eigenvectors are taken for basis. For a
        fractional value (in between), a fraction of high and low 
        eigen-basis vectors are used.
        This is relevant only if 'pca_dim_reduce' is not None (if PCA
        is being used).
    """
    # Use Residual based Global Descriptors
    use_residual: Literal[1, 0] = bool(0)
    # Number of Clusters for Residual based Global Descriptors
    num_clusters: int = 4
    # Cache file name (to save)
    save_all_descs: Union[None, Path] = None
    """
        All descriptors are saved in this file. If None, descriptors
        aren't saved. Parameter `cache_dir` should also not be None.
        File name shouldn't end with the '.npy' extension (it'll be 
        added). Name can contain '/' for directories.
    """


# %%
# Build cache of all images in the dataset
def build_cache(largs: LocalArgs, vpr_dl: DataLoader, model: Clip,vpr_dis_dl: DataLoader = None) \
        -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
        Builds the cache files, and returns the descriptors for the
        database and queries (along with positives_per_query)
        
        Parameters:
        - largs:    Local arguments for this file
        - vpr_dl:   DataLoader for retrieving the images
        - model:    The MixVPR model
        
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
        with torch.no_grad():
            # res = model.encode_image(img_batch, ci=int(ind_batch))
            res = model(img_batch.to(device))
        full_res.append(res)
    full_res = torch.concat(full_res, dim=0)
    full_res = full_res.detach().cpu()
    db_num = vpr_dl.dataset.database_num
    database_descs = full_res[:db_num]
    queries_descs = full_res[db_num:]
    print(f"Database descriptors shape: {database_descs.shape}")
    print(f"Query descriptors shape: {queries_descs.shape}")
    if largs.save_all_descs is not None and \
            largs.prog.cache_dir is not None:
        sf = f"{largs.prog.cache_dir}/{largs.save_all_descs}.npy"
        sf = os.path.realpath(os.path.expanduser(sf))
        base_d = os.path.dirname(sf)
        if os.path.isdir(base_d):
            print(f"Directory {base_d} exists. Overwriting!")
        else:
            os.makedirs(base_d)
        np.save(sf, full_res.numpy())
        print(f"Saved all descriptors to '{sf}'")
    #VPAir Condition
    if vpr_dis_dl is not None:
        try:
            full_dis_res = []
            for batch in tqdm(vpr_dis_dl):
                img_batch, ind_batch = batch[0], batch[1]
                assert img_batch.shape[0] == 1, "Batch size should be 1"
                with torch.no_grad():
                    # res = model.encode_image(img_batch, ci=int(ind_batch))
                    res = model(img_batch.to(device))
                full_dis_res.append(res)
            full_dis_res = torch.concat(full_dis_res, dim=0)
            full_dis_res = full_dis_res.detach().cpu()
            dis_db_num = vpr_dis_dl.dataset.database_num
            database_descs = torch.concatenate((database_descs,full_dis_res),0)
            print(f"Database with distractors shape: {database_descs.shape}")
        except RuntimeError as err:
            print(f"Experienced runtime error: {err}")
            print("Not using distractors")

    return database_descs, queries_descs, \
        vpr_dl.dataset.soft_positives_per_query


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
        if not largs.use_residual:
            qimgs_dir = f"{largs.prog.cache_dir}/experiments/"\
                        f"{largs.exp_id}/qualitative_retr"
        else:
            qimgs_dir = f"{largs.prog.cache_dir}/experiments/"\
                        f"{largs.exp_id}/qualitative_retr_residual_nc"\
                        f"{largs.num_clusters}"
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
            save_path = f"{qimgs_dir}/Q_{i_qu}_Top_"\
                        f"{largs.qual_num_rets}.png"
            fig.savefig(save_path)
            plt.close(fig)
            if largs.prog.use_wandb and largs.prog.wandb_save_qual:
                wandb.log({"Qual_Results": wandb.Image(save_path)})
    if use_percentage:
        for k in recalls:
            recalls[k] /= len(indices)  # As a percentage of queries
    return recalls


def get_enhanced_residual_vector(descriptors, cluster_centroids):
    """
        Get enhanced global descriptors based on residuals
    """
    num_desc = descriptors.shape[0]
    desc_dim = descriptors.shape[1]
    
    num_clusters = cluster_centroids.shape[0]
    
    residuals = torch.zeros(num_desc,(desc_dim*(num_clusters)))
    
    # slower implementation but uses lesser mem for large vector dims
    for c in range(num_clusters):
        cur_residuals = descriptors - cluster_centroids[c]
        residuals[:,((c)*desc_dim):((c+1)*desc_dim)] = F.normalize(
            cur_residuals, p=2.0)
    
    residuals = F.normalize(residuals, p=2.0, dim=-1)
    
    return residuals

# %%
# Main function
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything()
    
    if largs.prog.use_wandb:
        # Launch WandB
        wandb.init(project=largs.prog.wandb_proj, 
                entity=largs.prog.wandb_entity, config=largs, 
                group=largs.prog.wandb_group,
                name=largs.prog.wandb_run_name)
    
    print("------------------ VPR Model ------------------")
    model = VPRModel(backbone_arch='resnet50', 
                 layers_to_crop=[4],
                 agg_arch='MixVPR',
                 agg_config={'in_channels' : 1024,
                             'in_h' : 20,
                             'in_w' : 20,
                             'out_channels' : 1024,
                             'mix_depth' : 4,
                             'mlp_ratio' : 1,
                             'out_rows' : 4},
                )#.cuda()

    state_dict = torch.load(str(largs.ckpt_file))
    model.load_state_dict(state_dict)
    model.eval().cuda()

    print("-------------- VPR model loaded --------------")

    print("-------- Generating Global Descriptors --------")
    datasets_dir = largs.prog.data_vg_dir
    dataset_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {datasets_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset split: {largs.data_split}")
    # vpr_ds = BaseDataset(largs.bd_args, datasets_dir, dataset_name, 
    #                     largs.data_split)


    if dataset_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)
    elif dataset_name=="Oxford":
        vpr_ds = Oxford(datasets_dir,use_mixVPR=True)
    elif dataset_name=="Oxford_25m":
        vpr_ds = Oxford(datasets_dir,use_mixVPR=True,override_dist=25)
    elif dataset_name=="gardens":
        vpr_ds = Gardens(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)    
    elif dataset_name.startswith("hawkins"):
        vpr_ds = Hawkins(largs.bd_args,datasets_dir,"hawkins_long_corridor",largs.data_split,use_mixVPR=True)
    elif dataset_name=="VPAir":
        vpr_ds = VPAir(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)
        vpr_dis_ds = VPAir_Distractor(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)
    elif dataset_name=="laurel_caverns":
        vpr_ds = Laurel(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)
    elif dataset_name=="eiffel":
        vpr_ds = Eiffel(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)
    elif dataset_name.startswith("Tartan_GNSS"):
        vpr_ds = Aerial(largs.bd_args,datasets_dir,dataset_name,largs.data_split,use_mixVPR=True)
    else:
        vpr_ds = BaseDataset(largs.bd_args, datasets_dir, dataset_name, 
                            largs.data_split,use_mixVPR=True)

    if dataset_name=="VPAir":
        vpr_dl = DataLoader(vpr_ds, largs.batch_size, pin_memory=True, 
                            shuffle=False)
        vpr_dis_dl = DataLoader(vpr_dis_ds, largs.batch_size, pin_memory=True, 
                            shuffle=False)
        db_descs, qu_descs, pos_pq = build_cache(largs, vpr_dl, model,vpr_dis_dl)
    else:
        vpr_dl = DataLoader(vpr_ds, largs.batch_size, pin_memory=True, 
                            shuffle=False)
        db_descs, qu_descs, pos_pq = build_cache(largs, vpr_dl, model)

    # Normalize the descriptors
    ndb_descs = F.normalize(db_descs, p=2, dim=1)
    nqu_descs = F.normalize(qu_descs, p=2, dim=1)
    print("-------- Global descriptors generated --------")
    
    if largs.use_residual:
        print("-------- Generating Residual based Global Descriptors --------")
        print(f"-------- Num of Clusters: {largs.num_clusters} --------")
        kmeans = fpk.KMeans(n_clusters=largs.num_clusters, mode='cosine')
        labels = kmeans.fit_predict(ndb_descs)
        
        ndb_descs = get_enhanced_residual_vector(ndb_descs, kmeans.centroids)
        nqu_descs = get_enhanced_residual_vector(nqu_descs, kmeans.centroids)
        print("-------- Residual based Global descriptors generated --------")
    
    # Convert Torch tensors to Numpy arrays
    ndb_descs = ndb_descs.numpy()
    nqu_descs = nqu_descs.numpy()
    norm_descs = lambda x: x/np.linalg.norm(x, axis=-1, keepdims=True)
    
    if largs.pca_dim_reduce is not None:
        print("--------------- Applying PCA ---------------")
        n_original, n_down = ndb_descs.shape[1], largs.pca_dim_reduce
        print(f"Reducing from {n_original} to {n_down} dimensions")
        down_db_descs, down_qu_descs = reduce_pca(ndb_descs, 
                nqu_descs, n_down, largs.pca_low_factor)
        ndb_descs = norm_descs(down_db_descs)
        nqu_descs = norm_descs(down_qu_descs)
        print("------------ PCA applied to descriptors ------------")
    
    print("----------- FAISS Search started -----------")
    recalls = get_recalls(largs, ndb_descs, nqu_descs, pos_pq, vpr_dl,
            use_gpu=False)  # Use CPU FAISS for now
    print("------------ FAISS Search ended ------------")
    
    print("----------------- Results -----------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Experiment-ID": str(largs.exp_id),
        "Cache-dir": str(largs.prog.cache_dir),
        "Dataset-name": str(dataset_name),
        "Timestamp": str(ts),
        "Agg-Method": "Global",
        "FAISS-metric": largs.faiss_method,
    }
    print("Results:")
    for k in results:
        print(f"- {k}: {results[k]}")
    print("- Recalls:")
    for tk in recalls.keys():
        results[f"R@{tk}"] = recalls[tk]
        print(f"  - R@{tk}: {recalls[tk]:.5f}")
    
    if largs.prog.use_wandb:
        # Log to Wandb
        wandb.log(results)
        for tk in recalls:
            wandb.log({"Recall-All": recalls[tk]}, step=int(tk))
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
        if not largs.use_residual:
            save_res_file = f"{save_res_file}/results_{ts}.gz"
        else:
            save_res_file = f"{save_res_file}/results_residual_nc"\
                            f"{largs.num_clusters}_{ts}.gz"
        print(f"Saving result in: {save_res_file}")
        joblib.dump(results, save_res_file)
    else:
        print("Not saving results")
    print("--------------------- END ---------------------")


# Main entrypoint
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
# Experimental section

# # %%
# largs = LocalArgs()
# model = Clip(largs.clip_impl, largs.clip_backbone, 
#         largs.clip_pretrained, use_caching=largs.exp_id, 
#         base_cache_dir=largs.prog.cache_dir, device=device)


# # %%
# datasets_dir = largs.prog.data_vg_dir
# dataset_name = largs.prog.vg_dataset_name
# print(f"Dataset directory: {datasets_dir}")
# print(f"Dataset name: {dataset_name}")
# print(f"Dataset split: {largs.data_split}")
# vpr_ds = BaseDataset(largs.bd_args, datasets_dir, dataset_name, 
#                     largs.data_split)
# vpr_dl = DataLoader(vpr_ds, largs.batch_size, pin_memory=True, shuffle=False)
# db_descs, qu_descs, pos_pq = build_cache(largs, vpr_dl, model)

# %%
