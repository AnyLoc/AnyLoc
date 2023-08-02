# Contrastive learning over DINO VLAD features
"""
    DEPRECATED: Ignore this script
    
    TODO: Add support for `vg_bench` type in `pos_margin` in
    class DinoVLADEmbeddingDataset.
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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import einops as ein
from dino_extractor import ViTExtractor
import numpy as np
import tyro
from torchinfo import summary
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import joblib
from dataclasses import dataclass, field
from typing import Union, Literal, List, Tuple
import time
import traceback
# Library modules
from utilities import VLAD, seed_everything, get_top_k_recall
from configs import ProgArgs, prog_args, BaseDatasetArgs, device
from dvgl_benchmark.datasets_ws import BaseDataset


# %%
@dataclass
class LocalArgs:
    # Program arguments
    prog: ProgArgs = ProgArgs(vg_dataset_name="17places", 
            wandb_proj="Dino-Descs", 
            wandb_group="Contrastive-Dino-VLAD")
    # TODO: Use a different experiment ID
    exp_id: Union[bool,str,None] = "dino_contrastive/test_2"
    """
        Experiment ID for caching. If None, then no caching is done.
        If True, then cache is generated in the `cache_dir` in `prog`.
        If False, no cache is generated. If a string, then cache is
        generated in the `cache_dir`/experiments/`exp_id` folder.
    """
    batch_size: int = 200
    """
        Batch size for training.
    """
    num_epochs: int = 5
    """
        Number of epochs for training.
    """
    val_every: int = 1
    """
        Frequency for validation. If 0, then no validation is done.
        Validation involves: Getting recall at `top_k_vals` in the
        validation (test) set. Currently only supports `vpr_bench`.
    """
    ckpt_every: int = 5
    """
        Frequency for saving checkpoints. If 0, then no checkpoints
        are saved during training. The file name is of the form
        `model_e{epoch+1}.pth`. This is not saved if `cache_dir` is
        not set (i.e. `exp_id` is None or False).
        The final model is saved as `model_final.pth` regardless.
    """
    lr: float = 1e-3
    """
        Learning rate for training.
    """
    loss_temp: float = 1.0
    """
        Temperature for the softmax in the loss function.
    """
    # Model parameters
    model_type: Literal["dino_vits8", "dino_vits16", "dino_vitb8", 
            "dino_vitb16", "vit_small_patch8_224", 
            "vit_small_patch16_224", "vit_base_patch8_224", 
            "vit_base_patch16_224"] = "dino_vits8"
    """
        Model for Dino to use as the base model.
    """
    out_dim: int = 512
    """
        Output dimension of the MLP model.
    """
    hidden_dim: int = 1024
    """
        Hidden dimension of the MLP model.
    """
    # Contrastive learning parameters
    num_pos: int = 2
    """
        Number of positives to mine for each index (anchor).
    """
    num_neg: int = 5
    """
        Number of negatives to mine for each index (anchor).
    """
    pos_margin: Union[int,float] = 2
    """
        Positive margin for mining positives (from the given index).
        If the dataset is of `vpr_bench` type (`17places` or such),
        then this is the + or - margin from the index (anchor).
        If the dataset is of `vg_bench` type (`st_lucia` or such),
        then this is the distance (in meters) from the index.
        > **Note**: `vg_bench` is not supported (as of now).
    """
    sub_sample_db_vlad: int = 2
    """
        Sub-sampling of database images for estimation of VLAD cluster
        centers.
    """
    sub_sample_db: int = 1
    """
        Sub-sampling for database images for the VPR setting (during
        validation only)
    """
    sub_sample_qu: int = 1
    """
        Sub-sampling for query images for the VPR setting (during
        validation only)
    """
    # Dino parameters
    down_scale_res: Tuple[int, int] = (224, 298)
    """
        Downscale resolution for DINO features (before extracting).
    """
    desc_layer: int = 11
    """
        Layer from which to extract DINO features.
    """
    desc_facet: Literal["key", "query", "value", "token"] = "key"
    """
        Facet for extracting dino descriptors.
    """
    desc_bin: bool = False
    """
        Apply log binning to the descriptor.
    """
    num_clusters: int = 64
    """
        Number of clusters for VLAD descriptor.
    """
    top_k_vals: List[int] = field(default_factory=lambda: \
                                [1, 5, 10, 20])
    """
        Top-k values for recall calculation in validation phase.
    """


# %%
# --------------------- Classes ---------------------
# PyTorch Dataset for Dino + VLAD in contrastive setting
class DinoVLADEmbeddingDataset(Dataset):
    """
        Dataset for mining positives and negatives for VLAD 
        descriptors built over Dino features.
    """
    def __init__(self, db: BaseDataset, dino_extractor: ViTExtractor,
                trained_vlad: VLAD, num_pos: int=1, num_neg: int=5, 
                pos_inds: int=2, cache_dir: Union[Path, None]=None,
                img_size = (224, 298), dino_layers = 11, 
                dino_facet = "key") \
                -> None:
        """
            Parameters:
            - db:       Dataset object containing image paths.
            - dino_extractor:   DINO extractor object.
            - trained_vlad:     A trained VLAD object for extracting
                                VLAD descriptors.
            - num_pos:  Number of positives to mine for each index.
            - num_neg:  Number of negatives to mine for each index.
            - pos_inds:     Number of indices (above and below) to 
                            consider for mining positives.
            - cache_dir:    Directory to cache the DINO features. If
                            None, then no caching is done.
            - img_size:     Image size to resize the images to before
                            extracting DINO features.
            - dino_layers:  Layer from which to extract DINO features.
            - dino_facet:   Facet for DINO features.
        """
        super().__init__()
        # Arguments
        self.db: BaseDataset = db
        self.pos_inds: int = pos_inds
        self.num_pos: int = num_pos
        self.num_neg: int = num_neg
        # Get the paths for all images
        self.paths: List[str] = db.database_paths
        self.gt_pos: List[List[int]] = []
        """
            Ground truth positives for each index. It has N lists, 
            with each list having M indices (for positive elements).
            Value gt_pos[i] is the indices in paths that are a 
            positive retrieval
        """
        if self.db.vprbench:
            assert len(self.paths) > 2 * self.pos_inds, "Not enough "\
                    "images to mine positives and negatives"
            def get_pos_inds(i: int) -> List[int]:
                """
                    Get ground truth positives for self.paths[i]
                """
                st_i = i - self.pos_inds
                en_i = i + self.pos_inds
                if st_i < 0:
                    st_i = 0
                    en_i = st_i + 2 * self.pos_inds
                if en_i >= len(self.paths):
                    en_i = len(self.paths) - 1
                    st_i = en_i - 2 * self.pos_inds
                return list(range(st_i, en_i + 1))
            print("Mining ground truth positives")
            for i in tqdm(range(len(self.paths))):
                self.gt_pos.append(get_pos_inds(i))
        else:
            raise NotImplementedError("Only vpr_bench datasets are "\
                    "supported (for now)")
        self.paths: np.ndarray = np.array(self.paths)
        # Make the Dino VLAD + caching arrangement
        self.cdir = cache_dir
        if self.cdir is not None:
            if not os.path.isdir(self.cdir):
                os.makedirs(self.cdir)
                print(f"Created cache directory: {self.cdir}")
            else:
                print("Warning: Cache directory already exists")
        else:
            print("No caching will be done")
        self.dino = dino_extractor
        self.vlad = trained_vlad
        if type(img_size) == tuple:
            self.img_size = img_size
        else:   # Square image
            self.img_size = (img_size, img_size)
        self.dino_layers = dino_layers
        self.dino_facet = dino_facet
    
    def __len__(self) -> int:
        return len(self.paths)
    
    @torch.no_grad()
    def get_dino_vlad(self, index: int) -> torch.Tensor:
        """
            Extract the Dino VLAD features and cache them if not 
            already cached (and if applicable).
            
            Parameters:
            - index:        The index for retrieval (in self.paths)
            
            Returns:
            - dino_vlad:    The DINO VLAD features for the image at
                            self.paths[index]. Device is the same as
                            the device of the trained VLAD centers.
        """
        # Check if cached (and retrieve)
        if self.cdir is not None:
            cpath = os.path.join(self.cdir, f"{index}.pt")
            if os.path.isfile(cpath):
                return torch.load(cpath)\
                        .to(self.vlad.c_centers.device)
        # Extract the features
        img = self.db[index][0]
        img = ein.rearrange(img, "c h w -> 1 c h w")
        img = F.interpolate(img, self.img_size).to(self.dino.device)
        desc = self.dino.extract_descriptors(img, self.dino_layers,
                self.dino_facet, bin=False) # [1, 1, num_desc, d_dim]
        desc = ein.rearrange(desc, "1 1 n_d d_dim -> n_d d_dim")
        # Build VLAD
        dino_vlad = self.vlad.generate(desc\
                .to(self.vlad.c_centers.device))
        # Cache if applicable
        if self.cdir is not None:
            torch.save(dino_vlad.detach().cpu(), 
                    os.path.join(self.cdir, f"{index}.pt"))
        return dino_vlad
    
    @torch.no_grad()
    def build_dino_vlad_cache(self):
        """
            Build the Dino VLAD cache for all images in the dataset.
            This is not necessary, but can be useful if you want to
            generate the entire cache at once and read it in batches.
            By default (if you don't use this), the cache is built and
            used on the fly.
        """
        assert self.cdir is not None, "No cache directory specified"
        for i in tqdm(range(len(self))):
            self.get_dino_vlad(i)
    
    def __getitem__(self, index: int) -> dict:
        """
            Returns dictionary with the following keys:
            - "anchor": torch.Tensor of shape (1, D)
            - "pos":    torch.Tensor of shape (num_pos, D)
            - "neg":    torch.Tensor of shape (num_neg, D)
        """
        anchor = index
        anchor_i = self.paths[anchor]    # Image path
        anchor_vlad = self.get_dino_vlad(anchor)
        anchor_vlad = ein.rearrange(anchor_vlad, "d -> 1 d")
        pos = self.gt_pos[index]
        neg = [i for i in range(len(self.paths)) if i not in pos]
        pos = np.random.choice(pos, self.num_pos, replace=False)
        neg = np.random.choice(neg, self.num_neg, replace=False)
        pos_i = self.paths[pos].tolist()
        neg_i = self.paths[neg].tolist()
        pos_vlads = torch.stack([self.get_dino_vlad(i) for i in pos])
        neg_vlads = torch.stack([self.get_dino_vlad(i) for i in neg])
        return {
            "anchor": anchor_vlad,
            "pos": pos_vlads,
            "neg": neg_vlads
        }


# Create a shallow MLP
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


# %%
# --------------------- Functions ---------------------
# Loss function
def contrastive_loss(emb, pos, neg, temp=1.0):
    """
        Parameters:
        - emb:  torch.Tensor of shape (B, 1, D)
                VLAD Embedding of anchor image
        - pos:  torch.Tensor of shape (B, num_pos, D)
                VLAD Embeddings of positive images
        - neg:  torch.Tensor of shape (B, num_neg, D)
                VLAD Embeddings of negative images
        - temp: float
                Temperature for contrastive loss (softmax)
        
        Returns:
        - loss: torch.Tensor of shape (1,) (scalar)
                If batch is given, it is averaged over the batch
    """
    sim_emb_pos = F.cosine_similarity(emb, pos, dim=-1)
    sim_emb_neg = F.cosine_similarity(emb, neg, dim=-1)
    loss = -torch.log(torch.exp(sim_emb_pos / temp).sum(dim=-1) \
                    / torch.exp(sim_emb_neg / temp).sum(dim=-1))
    return loss.mean()


# Validation loop
@torch.no_grad()
def validation_step(largs: LocalArgs, net: MLP, db: BaseDataset, 
            dino: ViTExtractor, vlad: VLAD, verbose: bool=False):
    """
        Perform one validation step and return recall.
        
        Parameters:
        - largs:        Program argument
        - net:          MLP network for reducing VLAD resolution
        - db:           BaseDataset for loading
        - dino:         ViTExtractor for extracting Dino features
        - vlad:         For generating VLAD features
        - verbose:      Print progress and tqdm bar
        
        Returns a dictionary with the following keys:
        - "dists":       Distances with shape (num_qu, max(top_k))
        - "recalls":     Recall dictionary {k: recall@k}
        - "indices":     Indices with shape (num_qu, max(top_k))
        - "db_descs":    Database descriptors with shape (num_db, D)
        - "qu_descs":    Query descriptors with shape (num_qu, D)
    """
    num_db = db.database_num
    ds_len = len(db)
    # Database descriptors
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    db_descs = []
    for i in tqdm(db_indices, disable=(not verbose)):
        img = db[i][0]
        img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
        img = F.interpolate(img, largs.down_scale_res)
        # Dino descriptor - [1, 1, num_desc, dino_dim]
        desc = dino.extract_descriptors(img, layer=largs.desc_layer, 
                facet=largs.desc_facet, bin=largs.desc_bin)
        # VLAD global descriptor
        vl = vlad.generate(desc.squeeze().to(vlad.c_centers.device))
        # MLP to down-size it
        desc = net(vl.to(device))
        db_descs.append(desc)
    db_descs = torch.stack(db_descs)
    # Query descriptors
    qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    qu_descs = []
    for i in tqdm(qu_indices, disable=(not verbose)):
        img = db[i][0]
        img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
        img = F.interpolate(img, largs.down_scale_res)
        # Dino descriptor - [1, 1, num_desc, dino_dim]
        desc = dino.extract_descriptors(img, layer=largs.desc_layer, 
                facet=largs.desc_facet, bin=largs.desc_bin)
        # VLAD global descriptor
        vl = vlad.generate(desc.squeeze().to(vlad.c_centers.device))
        # MLP to down-size it
        desc = net(vl.to(device))
        qu_descs.append(desc)
    qu_descs = torch.stack(qu_descs)
    # Get recalls
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
            db_descs.to("cpu"), qu_descs.to("cpu"), 
            db.get_positives(),  sub_sample_db=largs.sub_sample_db,
            sub_sample_qu=largs.sub_sample_qu)
    return {
        "dists": dists,
        "indices": indices,
        "recalls": recalls,
        "db_descs": db_descs,
        "qu_descs": qu_descs
    }


# %%
# Main function
def main(largs: LocalArgs):
    seed_everything()
    
    # Caching and wandb
    cache_dir = None
    if largs.exp_id == True:
        cache_dir = f"{largs.prog.cache_dir}"
    elif type(largs.exp_id) == str:
        cache_dir = f"{largs.prog.cache_dir}/experiments/"\
                    f"{largs.exp_id}"
    if largs.prog.use_wandb:
        wandb_run = wandb.init(project=largs.prog.wandb_proj, 
                entity=largs.prog.wandb_entity, 
                group=largs.prog.wandb_group, config=largs)
    
    # Build VLAD cluster centers by loading dataset
    print("Building VLAD cluster centers")
    bd_args = BaseDatasetArgs(
            val_positive_dist_threshold=largs.pos_margin)
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}")
    # Dataset and extractor
    db = BaseDataset(bd_args, ds_dir, ds_name, "test")
    dino = ViTExtractor(largs.model_type, 4, device=device)
    vlad = VLAD(largs.num_clusters)
    num_db = db.database_num
    full_db_vlad = []   # List of [1, num_descs, d_dim] tensors
    db_indices = np.arange(0, num_db, largs.sub_sample_db_vlad)
    for i in tqdm(db_indices):
        img = db[i][0]
        img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
        img = F.interpolate(img, largs.down_scale_res)
        with torch.no_grad():
            desc = dino.extract_descriptors(img,
                    layer=largs.desc_layer, facet=largs.desc_facet,
                    bin=largs.desc_bin) # [1, 1, num_descs, d_dim]
        full_db_vlad.append(desc.squeeze().cpu())
    full_db_vlad = torch.stack(full_db_vlad)    # [n_img, n_desc, d]
    print(f"DB (for VLAD) shape: {full_db_vlad.shape}")
    # Build VLAD cluster centers
    _start = time.time()
    train_desc = ein.rearrange(full_db_vlad, 
                                "n d d_dim -> (n d) d_dim")
    vlad.fit(train_desc)
    print(f"VLAD training time: {time.time() - _start:.3f} seconds")
    print(f"VLAD desc. dim: {largs.num_clusters * vlad.desc_dim}")
    
    # Build DinoVLAD dataset
    ds = DinoVLADEmbeddingDataset(db, dino, vlad, 
        num_pos=largs.num_pos, num_neg=largs.num_neg,
        pos_inds=largs.pos_margin, 
        cache_dir=f"{cache_dir}/vlad_cache" if cache_dir else None)
    if cache_dir:
        print("Building Dino-VLAD cache in advance")
        ds.build_dino_vlad_cache()
    else:
        print("Not building cache as cache_dir is None or False")
    
    # Build model and optimizer
    net = MLP(largs.num_clusters * vlad.desc_dim, largs.out_dim, 
            largs.hidden_dim).to(device)
    in_size = (largs.batch_size, 1, largs.num_clusters*vlad.desc_dim)
    print(f"Built MLP, showing properties for input: {in_size}")
    summary(net, input_size=in_size)
    opt = torch.optim.Adam(net.parameters(), lr=largs.lr)
    
    # Train model
    print("Training model")
    _start = time.time()
    for epoch in tqdm(range(largs.num_epochs), position=0, 
                leave=True):
        
        # Validation code for an epoch
        def validate_epoch(e: int):
            # 'e' value used only for tSNE saving
            if not db.vprbench:
                print("Skipping validation, only vpr_bench supported")
            else:
                # print("Getting top-k validation results")
                validation_res = validation_step(largs, net, db, dino,
                        vlad)
                recalls = validation_res["recalls"]
                if largs.prog.use_wandb:
                    _rec = dict()
                    for r in recalls:
                        _rec[f"R@{r}"] = recalls[r]
                    wandb.log(_rec, step=e)
                else:
                    print("Recalls: ", end="")
                    for r in recalls:
                        print(f"{r}: {recalls[r]:.3f} ", end="")
                # Save tSNE plot for database and query
                if cache_dir is not None:
                    db_descs = validation_res["db_descs"]
                    qu_descs = validation_res["qu_descs"]
                    res1 = TSNE(2, learning_rate="auto", init="pca", 
                            perplexity=30, n_iter=1000)\
                            .fit_transform(db_descs.to("cpu").numpy())
                    res2 = TSNE(2, learning_rate="auto", init="pca", 
                            perplexity=30, n_iter=1000)\
                            .fit_transform(qu_descs.to("cpu").numpy())
                    fig = plt.figure(figsize=(10, 5))
                    gs = fig.add_gridspec(1, 2)
                    ax = fig.add_subplot(gs[0, 0])
                    ax.set_title("Database")
                    ax.scatter(res1[:, 0], res1[:, 1])
                    ax.tick_params(left=False, right=False, 
                        labelleft=False, labelbottom=False, 
                        bottom=False)
                    ax = fig.add_subplot(gs[0, 1])
                    ax.set_title("Queries")
                    ax.scatter(res2[:, 0], res2[:, 1])
                    ax.tick_params(left=False, right=False, 
                        labelleft=False, labelbottom=False, 
                        bottom=False)
                    fig.set_tight_layout(True)
                    if largs.prog.use_wandb and \
                            largs.prog.wandb_save_qual:
                        wandb.log({"tsne": wandb.Image(fig)}, step=e)
                    fig.savefig(f"{cache_dir}/tsne_e{e}.png", 
                        dpi=300)
                    plt.close(fig)
        
        if epoch == 0:
            validate_epoch(0)
        
        # Train for all batches in the epoch
        dl = DataLoader(ds, batch_size=largs.batch_size)
        loss_avg, num_steps = 0.0, 0
        for bidx, batch in enumerate(dl):
            opt.zero_grad()
            emb_anchor = batch["anchor"].to(device)
            emb_pos = batch["pos"].to(device)
            emb_neg = batch["neg"].to(device)
            # Forward pass
            out_anchor = net(emb_anchor)
            out_pos = net(emb_pos)
            out_neg = net(emb_neg)
            # Compute loss
            loss = contrastive_loss(out_anchor, out_pos, out_neg, 
                                    largs.loss_temp)
            # Backward pass
            loss.backward()
            opt.step()
            loss_avg += loss.item()
            num_steps += 1
        
        if largs.prog.use_wandb:
            wandb.log({"loss": loss_avg/num_steps}, step=epoch+1)
        else:
            print(f"Epoch {epoch} - loss: {loss_avg/num_steps:.5f}")
        
        # Check for validation
        if largs.val_every != 0 and (epoch % largs.val_every == 0 or \
                    epoch == largs.num_epochs - 1):
            validate_epoch(epoch+1) # +1 because epoch has ended
        
        # Checkpoint
        if cache_dir is not None and (epoch % largs.ckpt_every == 0):
            torch.save({"epoch": epoch, 
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss_avg / num_steps
                }, f"{cache_dir}/model_e{epoch+1}.pth")
    # Final model
    if cache_dir is not None:
        torch.save({"epoch": epoch, 
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss_avg / num_steps
                }, f"{cache_dir}/model_final.pth")
        print(f"Model saved to '{cache_dir}/model_final.pth'")
    else:
        print("No cache_dir, saving checkpoint to model_final.pth")
        torch.save({"epoch": epoch, 
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss_avg / num_steps
                }, f"model_final.pth")
    
    # End program
    if largs.prog.use_wandb:
        wandb.finish()
    print(f"Training took {time.time() - _start:.3f} seconds")


if __name__ == "__main__" and (not "ipykernel" in sys.argv[0]):
    # Parse arguments
    args = tyro.cli(LocalArgs)
    print(f"Arguments: {args}")
    _start = time.time()
    try:
        main(args)
    except:
        print("Unhandled exception")
        traceback.print_exc()
    finally:
        print(f"Finished in {time.time() - _start:.3f} seconds")
        exit(0)


# %%
# Experiments
