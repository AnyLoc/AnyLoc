# Utility scripts
"""
"""

# %%
import numpy as np
import torch
from torch import nn
import transformers as hft
from torch.nn import functional as F
import einops as ein
import fast_pytorch_kmeans as fpk
import faiss
import faiss.contrib.torch_utils
import random
import os
from PIL import Image
from sklearn.decomposition import PCA
from typing import Union, List, Tuple, Literal

import matplotlib.pyplot as plt

# %% ---------------- Dataset Utilities ----------------
# Abstract class (parent) for all custom datasets
class CustomDataset:
    def __init__(self) -> None:
        # Required properties
        self.database_num = None    # Number of database items
        self.queries_num = None     # Number of queries
        self.soft_positives_per_query = None    # Soft pos per qu
    
    def get_image_paths(self):
        if hasattr(self, 'images_paths'):
            return self.images_paths
        else:
            raise NotImplementedError("Not handled!")
    
    def get_positives(self):
        if hasattr(self, 'soft_positives_per_query'):
            return self.soft_positives_per_query
        else:
            raise NotImplementedError("Not handled!")
    
    def get_image_relpaths(self, i: Union[int, List[int]]) \
            -> Union[List[str], str]:
        """
            Get the relative path of the image at index i (in the 
            dataset). Multiple  indices can be passed as a list (or 
            int-like array). This could be useful for caching.
            
            > Note: If images are at a level other than 2, then the
                variable _imgs_level should be initialized
        """
        indices = i
        if type(i) == int:
            indices = [i]
        img_paths = self.get_image_paths()
        s = 2
        if hasattr(self, '_imgs_level'):
            s = self._imgs_level
        rel_paths = ["/".join(img_paths[k].split("/")[-s:]) \
                        for k in indices]
        if type(i) == int:
            return rel_paths[0]
        return rel_paths
    
    def __getitem__(self, index):
        raise NotImplementedError("Not created!")
    
    def __len__(self):
        if hasattr(self, 'images_paths'):
            return len(self.get_image_paths())
        else:
            raise NotImplementedError("Not handled!")


# %% -------------------- Converter functions --------------------
# Convert to numpy
def to_np(x, ret_type=float) -> np.ndarray:
    """
        Converts 'x' to numpy object of `dtype` as 'ret_type'
        
        Parameters:
        - x:    An object
        
        Returns:
        - x_np:     A numpy array of dtype `ret_type`
    """
    x_np: np.ndarray = None
    if type(x) == torch.Tensor:
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x)
    x_np = x_np.astype(ret_type)
    return x_np


# Convert to PIL image
def to_pil_list(x) -> List[Image.Image]:
    """
        Converts the input 'x' object to a list of PIL Images 
        (assuming that 'x' is really an image or a batch of images). 
        You can pass a batch of shape [B, C, H, W] or shape 
        [B, H, W, C] and it returns a list of PIL Images. If 'x' is of
        shape [H, W, C] or [C, H, W], then the length of list is 1.
        
        Parameters:
        - x:    A single or a batch of images (channels first or last)
        
        Returns:
        - imgs_pil:     A list of PIL Images (length is the number of 
                        images in 'x')
    """
    if type(x) == Image.Image or \
            (type(x) == list and type(x[0]) == Image.Image):
        return x    # Passthrough
    else:
        x = to_np(x)
    if len(x.shape) == 3:
        x = x[np.newaxis, ...]  # Now len(x.shape) is 4
    imgs_pil = []
    for x_img in x:
        if x_img.shape[0] in [1, 3]:    # [C, H, W] format
            x_img = x_img.transpose(1, 2, 0)    # Now [H, W, C]
        # Normalize image
        x_norm = (x_img - x_img.min())/(x_img.max() - x_img.min())
        x_pil = Image.fromarray((x_norm * 255).astype(np.uint8))
        imgs_pil.append(x_pil)
    return imgs_pil


# %%
_VIT_FACETS = Literal["query", "key", "value", "token"]
class CosPlaceViTExtractFeatures:
    """
        Extract features from an intermediate layer in CosPlace.
    """
    def __init__(self, ckpt_path: str, layer: int, facet: _VIT_FACETS,
                use_cls: bool=False, norm_descs: bool=True,
                device="cpu") -> None:
        """
            Parameters:
            - ckpt_path: str    Checkpoint path
            - layer: int        Layer number
            - facet: str        Facet to use
            - use_cls: bool     If True, the CLS token is retained
            - norm_descs: bool  If True, normalize patch descriptors
            - device: Union[torch.device, str]
        """
        cfg = hft.ViTConfig()
        self.ckpt_path = ckpt_path
        assert os.path.isfile(self.ckpt_path), "Checkpoint not found"
        self.device = torch.device(device)
        self.model: nn.Module = hft.ViTModel(cfg)
        ckpt = torch.load(self.ckpt_path)
        res = self.model.load_state_dict(ckpt)
        print(f"Checkpoint loaded, result: {res}")
        self.model = self.model.eval().to(self.device)
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Create hook
        if self.facet == "token":
            self.hook_handle = self.model.encoder.layer[self.layer]\
                    .register_forward_hook(self\
                        ._generate_forward_hook(self.facet))
        elif self.facet == "key":
            self.hook_handle = self.model.encoder.layer[self.layer]\
                    .attention.attention.key.register_forward_hook(
                        self._generate_forward_hook(self.facet))
        elif self.facet == "query":
            self.hook_handle = self.model.encoder.layer[self.layer]\
                    .attention.attention.query.register_forward_hook(
                        self._generate_forward_hook(self.facet))
        elif self.facet == "value":
            self.hook_handle = self.model.encoder.layer[self.layer]\
                    .attention.attention.value.register_forward_hook(
                        self._generate_forward_hook(self.facet))
        else:
            raise ValueError(f"Invalid facet: {self.facet}")
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self, facet: _VIT_FACETS):
        def _forward_hook(module, inputs, output):
            if facet == "token":   # It's a tuple of len = 1
                self._hook_out = output[0]
            else:
                self._hook_out = output
        return _forward_hook
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img: torch.Tensor     Input image
        """
        with torch.no_grad():
            self._hook_out: torch.Tensor = None
            res = self.model(img)
            assert self._hook_out is not None, "No data from hook"
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        return res
    
    def __del__(self):
        self.hook_handle.remove()


# %% -------------------- Dino-v2 utilities --------------------
# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
                        "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]
class DinoV2ExtractFeatures:
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int, 
                facet: _DINO_FACETS="token", use_cls=False, 
                norm_descs=True, device: str = "cpu") -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        self.vit_type: str = dino_model
        self.dino_model: nn.Module = torch.hub.load(
                'facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2*d_len]
                else:
                    res = res[:, :, 2*d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None   # Reset the hook
        return res
    
    def __del__(self):
        self.fh_handle.remove()


# %% -------------- MAE Utilities (Position embedding) --------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. (only for this code block)
# Directly from: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



# %% -------------------- Recall Calculations --------------------
def get_top_k_recall(top_k: List[int], db: torch.Tensor, 
        qu: torch.Tensor, gt_pos: np.ndarray, method: str="cosine", 
        norm_descs: bool=True, use_gpu: bool=False, 
        use_percentage: bool=True, sub_sample_db: int=1, 
        sub_sample_qu: int=1) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
        Given a database and query (or queries), get the top 'k'
        retrievals (closest in database for each query) as indices (in
        database), distances, and recalls.
        
        Parameters:
        - top_k:    List of 'k' values for recall calculation. Eg:
                    `list(range(1, 11))`.
        - db:       Database descriptors of shape [n_db, d_dim].
        - qu:       Query descriptors of shape [n_qu, d_dim]. If only
                    one query (n_qu = 1), then shape [d_dim].
        - gt_pos:   Ground truth for retrievals. Should be object type
                    with gt_pos[i] having true database items 
                    (indices) for the query 'i' (in `qu`).
        - method:   Method for faiss search. In {'cosine', 'l2'}.
        - norm_descs:   If True, the descriptors are normalized in 
                        function.
        - use_gpu:  True if indexing (search) should be on GPU.
        - use_percentage:   If True, the recalls are returned as a
                            percentage (of queries resolved).
        - sub_sample_db:    Sub-sample database samples from the
                            ground truth 'gt_pos'
        - sub_sample_qu:    Sub-sample query samples from the ground
                            truth 'gt_pos'
        
        Returns:
        - distances:    The distances of queries to retrievals. The
                        shape is [n_qu, max(top_k)]. It is the 
                        distance (as specified in `method`) with the
                        database item retrieved (index in `indices`).
                        Sorted by distance.
        - indices:      Indices of the database items retrieved. The
                        shape is [n_qu, max(top_k)]. Sorted by 
                        distance.
        - recalls:      A dictionary with keys as top_k integers, and
                        values are the recall (number or percentage)
                        of correct retrievals for queries.
    """
    if len(qu.shape) == 1:
        qu = qu.unsqueeze(0)
    if norm_descs:
        db = F.normalize(db)
        qu = F.normalize(qu)
    D = db.shape[1]
    if method == "cosine":
        index = faiss.IndexFlatIP(D)
    elif method == "l2":
        index = faiss.IndexFlatL2(D)
    else:
        raise NotImplementedError(f"Method: {method}")
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0 , index)
    # Get the max(top-k) retrieval, then traverse list
    index.add(db)
    distances, indices = index.search(qu, max(top_k))
    recalls = dict(zip(top_k, [0]*len(top_k)))
    # print(qu.shape,indices.shape)
    for i_qu, qu_retr in enumerate(indices):
        for i_rec in top_k:
            # Correct database images (for the retrieval)
            """
                i_qu * sub_sample_qu
                    Sub-sampled queries (step)
                qu_retr[:i_rec] * sub_sample_db
                    Sub-sampled database (step in retrievals)
            """
            correct_retr = gt_pos[i_qu * sub_sample_qu]
            if np.any(np.isin(qu_retr[:i_rec] * sub_sample_db, 
                        correct_retr)):
                recalls[i_rec] += 1
    if use_percentage:
        for k in recalls:
            recalls[k] /= len(indices)
    return distances, indices, recalls


# %% --------------- Image processing functions ----------------
# Pad an image
def pad_img(img: np.ndarray, padding:int, color:tuple=(0, 0, 0)) \
        -> np.ndarray:
    """
        Pad an image with 'padding' along each side (height and width)
        and fill the padding with 'color'.
        
        Parameters:
        - img:  Image of shape [H, W, C=3] with channels as RGB (same
                as the 'color' channels)
        - padding:      Padding 'P' (int) for each dimension (applied 
                        on both ends of axis)
        - color:    The RGB color of the padding
        
        Returns:
        - _img:     Image of shape [H+2P, W+2P, C=3]
    """
    if type(color) == list:
        color = tuple(color)
    assert len(color) == 3, "Color should be (R, G, B) value"
    color = np.array(color)
    # ret_img = np.pad(img, [(padding, padding), (padding, padding), 
    #             (0, 0)], constant_values=[(color, color), 
    #                         (color, color), (0, 0)])
    ret_img = np.ones((img.shape[0] + 2*padding, 
                img.shape[1] + 2*padding, 3), np.uint8) * color
    ret_img[padding:-padding, padding:-padding] = img
    return ret_img.astype(img.dtype)


# %% ------------------- Utility functions -------------------
# Set a seed value
def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")

# PCA dimensionality reduction
def reduce_pca(train_descs: np.ndarray, test_descs: np.ndarray, 
            lower_dim:int, low_factor:float=0.0, fallback:int=256,
            svd_solver:str='full') -> Tuple[np.ndarray, np.ndarray]:
    """
        Reduce the dimensionality of the training and test dataset
        using Principal Component Analysis (implementation from 
        sklearn.decomposition). The test set is reduced by using the
        parameters (basis and mean) from the training set.
        
        Parameters:
        - train_descs:  Training descriptors of shape [n_tr, n_o]
        - test_descs:   Test set descriptors of shape [n_ts, n_o]
        - lower_dim:    The number of components in the PCA (after
                        dimensionality reduction), `l_dim`.
        - low_factor:   The percentage of (eigen)basis vectors to take
                        from the lower end of the eigenvectors (
                        having least eigenvalues). For example, if 30%
                        and `lower_dim` is 100, then 60 basis vectors
                        are taken from top eigenvalues and 30 basis
                        vectors are taken from the lowest eigenvalues.
        - fallback:     If `n_tr < l_dim` (lesser samples than final
                        components) for the training set and the
                        `low_factor` is non-zero, then the training
                        and test samples are first directly projected
                        to this lower dimension. They are then further
                        reduced using `lower_dim`. This must be higher
                        than `lower_dim` value (for it to make sense).
        - svd_solver:   The solver for scipy PCA module.
        
        Returns:
        - out_train_descs:  Training descriptors: [n_tr, l_dim]
        - out_test_descs:   Test descriptors: [n_ts, l_dim]
    """
    assert 0 <= low_factor <= 1
    out_train_descs: np.ndarray = None
    out_test_descs: np.ndarray = None
    if low_factor == 0.0:   # Direct downsample
        pca = PCA(lower_dim, svd_solver=svd_solver)
        out_train_descs = pca.fit_transform(train_descs)
        out_test_descs = pca.transform(test_descs)
    else:
        n_samples, n_components = train_descs.shape
        if n_samples < n_components:
            print(f"Too few samples, fallback to {fallback}d first")
            _train_descs = train_descs.copy()
            _test_descs = test_descs.copy()
            _all_descs = np.concatenate((_train_descs, _test_descs))
            pca = PCA(fallback, svd_solver=svd_solver)
            _all_descs_down = pca.fit_transform(_all_descs)
            train_descs = _all_descs_down[:n_samples]
            test_descs = _all_descs_down[len(train_descs):]
        _down = int(low_factor * lower_dim)
        _up = lower_dim - _down
        print(f"Up: {_up}, Down: {_down}")
        n_samples, n_components = train_descs.shape
        pca = PCA(n_components, svd_solver=svd_solver)
        pca.fit(train_descs)
        tf_pca = np.concatenate((pca.components_[:_up],
                    pca.components_[-_down:]))
        out_train_descs = (train_descs - pca.mean_) @ tf_pca.T
        out_test_descs = (test_descs - pca.mean_) @ tf_pca.T
    return out_train_descs, out_test_descs


# Concatenate descriptor distances (residuals) from cluster centers
def concat_desc_dists_clusters(cluster_centers: torch.Tensor, \
        descs: torch.Tensor) -> torch.Tensor:
    """
        Do concatenation of descriptor distances from the cluster 
        centers. Performs the following steps:
        1. Calculate the distance vector  of each descriptor from each
            cluster center. 
        2. Normalize this distance vector (for each descriptor's each
            cluster center measurement) like intra-normalization.
        3. Concatenate the cluster center distance vectors across all
            cluster centers.
        4. Normalize this concatenated vector (per descriptor).
        
        Parameters:
        - cluster_centers:      (k, d) shape cluster centers
        - descs:            (n, d) shape descriptors
        
        Returns:
        - ncat_vects:   (n, (k*d)) shape pooled descriptors
    """
    assert type(cluster_centers) == type(descs) == torch.Tensor
    # Difference of all descriptors with cluster centers: (n, k, d)
    all_dists = descs[:, None, :] - cluster_centers[None, ...]
    # Intra-cluster normalization (norm the last dimension)
    nall_dists = all_dists / all_dists.norm(dim=-1, keepdim=True)
    # Concatenate the individual descriptors into a long vector
    cat_vects = ein.rearrange(nall_dists, "n k d -> n (k d)")
    # Normalize the concatenated vectors: (n, (k*d))
    ncat_vects = cat_vects / cat_vects.norm(dim=-1, keepdim=True)
    return ncat_vects


# %% --------------------- Utility classes ---------------------
# VLAD global descriptor implementation
class VLAD:
    """
        An implementation of VLAD algorithm given database and query
        descriptors.
        
        Constructor arguments:
        - num_clusters:     Number of cluster centers for VLAD
        - desc_dim:         Descriptor dimension. If None, then it is
                            inferred when running `fit` method.
        - intra_norm:       If True, intra normalization is applied
                            when constructing VLAD
        - norm_descs:       If True, the given descriptors are 
                            normalized before training and predicting 
                            VLAD descriptors. Different from the
                            `intra_norm` argument.
        - dist_mode:        Distance mode for KMeans clustering for 
                            vocabulary (not residuals). Must be in 
                            {'euclidean', 'cosine'}.
        - vlad_mode:        Mode for descriptor assignment (to cluster
                            centers) in VLAD generation. Must be in
                            {'soft', 'hard'}
        - soft_temp:        Temperature for softmax (if 'vald_mode' is
                            'soft') for assignment
        - cache_dir:        Directory to cache the VLAD vectors. If
                            None, then no caching is done. If a str,
                            then it is assumed as the folder path. Use
                            absolute paths.
        
        Notes:
        - Arandjelovic, Relja, and Andrew Zisserman. "All about VLAD."
            Proceedings of the IEEE conference on Computer Vision and 
            Pattern Recognition. 2013.
    """
    def __init__(self, num_clusters: int, 
                desc_dim: Union[int, None]=None, 
                intra_norm: bool=True, norm_descs: bool=True, 
                dist_mode: str="cosine", vlad_mode: str="hard", 
                soft_temp: float=1.0, 
                cache_dir: Union[str,None]=None) -> None:
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = dist_mode
        self.vlad_mode = str(vlad_mode).lower()
        assert self.vlad_mode in ['soft', 'hard']
        self.soft_temp = soft_temp
        # Set in the training phase
        self.c_centers = None
        self.kmeans = None
        # Set the caching
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(
                    self.cache_dir))
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            else:
                print("Warning: Cache directory already exists: " \
                        f"{self.cache_dir}")
        else:
            print("VLAD caching is disabled.")
    
    def can_use_cache_vlad(self):
        """
            Checks if the cache directory is a valid cache directory.
            For it to be valid, it must exist and should at least
            include the cluster centers file.
            
            Returns:
            - True if the cache directory is valid
            - False if 
                - the cache directory doesn't exist
                - exists but doesn't contain the cluster centers
                - no caching is set in constructor
        """
        if self.cache_dir is None:
            return False
        if not os.path.exists(self.cache_dir):
            return False
        if os.path.exists(f"{self.cache_dir}/c_centers.pt"):
            return True
        else:
            return False
    
    def can_use_cache_ids(self, 
                cache_ids: Union[List[str], str, None],
                only_residuals: bool=False) -> bool:
        """
            Checks if the given cache IDs exist in the cache directory
            and returns True if all of them exist.
            The cache is stored in the following files:
            - c_centers.pt:     Cluster centers
            - `cache_id`_r.pt:  Residuals for VLAD
            - `cache_id`_l.pt:  Labels for VLAD (hard assignment)
            - `cache_id`_s.pt:  Soft assignment for VLAD
            
            The function returns False if cache cannot be used or if
            any of the cache IDs are not found. If all cache IDs are
            found, then True is returned.
            
            This function is mainly for use outside the VLAD class.
        """
        if not self.can_use_cache_vlad():
            return False
        if cache_ids is None:
            return False
        if isinstance(cache_ids, str):
            cache_ids = [cache_ids]
        for cache_id in cache_ids:
            if not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_r.pt"):
                return False
            if self.vlad_mode == "hard" and not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_l.pt") and not \
                        only_residuals:
                return False
            if self.vlad_mode == "soft" and not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_s.pt") and not \
                        only_residuals:
                return False
        return True
    
    # Generate cluster centers
    def fit(self, train_descs: Union[np.ndarray, torch.Tensor, None]):
        """
            Using the training descriptors, generate the cluster 
            centers (vocabulary). Function expects all descriptors in
            a single list (see `fit_and_generate` for a batch of 
            images).
            If the cache directory is valid, then retrieves cluster
            centers from there (the `train_descs` are ignored). 
            Otherwise, stores the cluster centers in the cache 
            directory (if using caching).
            
            Parameters:
            - train_descs:  Training descriptors of shape 
                            [num_train_desc, desc_dim]. If None, then
                            caching should be valid (else ValueError).
        """
        # Clustering to create vocabulary
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        # Check if cache exists
        if self.can_use_cache_vlad():
            print("Using cached cluster centers")
            self.c_centers = torch.load(
                    f"{self.cache_dir}/c_centers.pt")
            self.kmeans.centroids = self.c_centers
            if self.desc_dim is None:
                self.desc_dim = self.c_centers.shape[1]
                print(f"Desc dim set to {self.desc_dim}")
        else:
            if train_descs is None:
                raise ValueError("No training descriptors given")
            if type(train_descs) == np.ndarray:
                train_descs = torch.from_numpy(train_descs).\
                    to(torch.float32)
            if self.desc_dim is None:
                self.desc_dim = train_descs.shape[1]
            if self.norm_descs:
                train_descs = F.normalize(train_descs)
            self.kmeans.fit(train_descs)
            self.c_centers = self.kmeans.centroids
            if self.cache_dir is not None:
                print("Caching cluster centers")
                torch.save(self.c_centers, 
                        f"{self.cache_dir}/c_centers.pt")
    
    def fit_and_generate(self, 
                train_descs: Union[np.ndarray, torch.Tensor]) \
                -> torch.Tensor:
        """
            Given a batch of descriptors over images, `fit` the VLAD
            and generate the global descriptors for the training
            images. Use only when there are a fixed number of 
            descriptors in each image.
            
            Parameters:
            - train_descs:  Training image descriptors of shape
                            [num_imgs, num_descs, desc_dim]. There are
                            'num_imgs' images, each image has 
                            'num_descs' descriptors and each 
                            descriptor is 'desc_dim' dimensional.
            
            Returns:
            - train_vlads:  The VLAD vectors of all training images.
                            Shape: [num_imgs, num_clusters*desc_dim]
        """
        # Generate vocabulary
        all_descs = ein.rearrange(train_descs, "n k d -> (n k) d")
        self.fit(all_descs)
        # For each image, stack VLAD
        return torch.stack([self.generate(tr) for tr in train_descs])
    
    def generate(self, query_descs: Union[np.ndarray, torch.Tensor],
                cache_id: Union[str, None]=None) -> torch.Tensor:
        """
            Given the query descriptors, generate a VLAD vector. Call
            `fit` before using this method. Use this for only single
            images and with descriptors stacked. Use function
            `generate_multi` for multiple images.
            
            Parameters:
            - query_descs:  Query descriptors of shape [n_q, desc_dim]
                            where 'n_q' is number of 'desc_dim' 
                            dimensional descriptors in a query image.
            - cache_id:     If not None, then the VLAD vector is
                            constructed using the residual and labels
                            from this file.
            
            Returns:
            - n_vlas:   Normalized VLAD: [num_clusters*desc_dim]
        """
        residuals = self.generate_res_vec(query_descs, cache_id)
        # Un-normalized VLAD vector: [c*d,]
        un_vlad = torch.zeros(self.num_clusters * self.desc_dim)
        if self.vlad_mode == 'hard':
            # Get labels for assignment of descriptors
            if cache_id is not None and self.can_use_cache_vlad() \
                    and os.path.isfile(
                        f"{self.cache_dir}/{cache_id}_l.pt"):
                labels = torch.load(
                        f"{self.cache_dir}/{cache_id}_l.pt")
            else:
                labels = self.kmeans.predict(query_descs)   # [q]
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(labels, 
                            f"{self.cache_dir}/{cache_id}_l.pt")
            # Create VLAD from residuals and labels
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                # Sum of residuals for the descriptors in the cluster
                #  Shape:[q, c, d]  ->  [q', d] -> [d]
                cd_sum = residuals[labels==k,k].sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        else:       # Soft cluster assignment
            # Cosine similarity: 1 = close, -1 = away
            if cache_id is not None and self.can_use_cache_vlad() \
                    and os.path.isfile(
                        f"{self.cache_dir}/{cache_id}_s.pt"):
                soft_assign = torch.load(
                        f"{self.cache_dir}/{cache_id}_s.pt")
            else:
                cos_sims = F.cosine_similarity( # [q, c]
                        ein.rearrange(query_descs, "q d -> q 1 d"), 
                        ein.rearrange(self.c_centers, "c d -> 1 c d"), 
                        dim=2)
                soft_assign = F.softmax(self.soft_temp*cos_sims, 
                        dim=1)
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(soft_assign, 
                            f"{self.cache_dir}/{cache_id}_s.pt")
            # Soft assignment scores (as probabilities): [q, c]
            for k in range(0, self.num_clusters):
                w = ein.rearrange(soft_assign[:, k], "q -> q 1 1")
                # Sum of residuals for all descriptors (for cluster k)
                cd_sum = ein.rearrange(w * residuals, 
                            "q c d -> (q c) d").sum(dim=0)  # [d]
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        # Normalize the VLAD vector
        n_vlad = F.normalize(un_vlad, dim=0)
        return n_vlad
    
    def generate_multi(self, 
            multi_query: Union[np.ndarray, torch.Tensor, list],
            cache_ids: Union[List[str], None]=None) \
            -> Union[torch.Tensor, list]:
        """
            Given query descriptors from multiple images, generate
            the VLAD for them.
            
            Parameters:
            - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                            There are 'n_imgs' and each image has
                            'n_kpts' keypoints, with 'd' dimensional
                            descriptor each. If a List (can then have
                            different number of keypoints in each 
                            image), then the result is also a list.
            - cache_ids:    Cache IDs for the VLAD vectors. If None,
                            then no caching is done (stored or 
                            retrieved). If a list, then the length
                            should be 'n_imgs' (one per image).
            
            Returns:
            - multi_res:    VLAD descriptors for the queries
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate(q, c) \
                for (q, c) in zip(multi_query, cache_ids)]
        try:    # Most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:    # Otherwise numpy
                res = np.stack(res)
            except TypeError:
                pass    # Let it remain as a list
        return res
    
    def generate_res_vec(self, 
                query_descs: Union[np.ndarray, torch.Tensor],
                cache_id: Union[str, None]=None) -> torch.Tensor:
        """
            Given the query descriptors, generate a VLAD vector. Call
            `fit` before using this method. Use this for only single
            images and with descriptors stacked. Use function
            `generate_multi` for multiple images.
            
            Parameters:
            - query_descs:  Query descriptors of shape [n_q, desc_dim]
                            where 'n_q' is number of 'desc_dim' 
                            dimensional descriptors in a query image.
            - cache_id:     If not None, then the VLAD vector is
                            constructed using the residual and labels
                            from this file.
            
            Returns:
            - residuals:    Residual vector: shape [n_q, n_c, d]
        """
        assert self.kmeans is not None
        assert self.c_centers is not None
        # Compute residuals (all query to cluster): [q, c, d]
        if cache_id is not None and self.can_use_cache_vlad() and \
                os.path.isfile(f"{self.cache_dir}/{cache_id}_r.pt"):
            residuals = torch.load(
                    f"{self.cache_dir}/{cache_id}_r.pt")
        else:
            if type(query_descs) == np.ndarray:
                query_descs = torch.from_numpy(query_descs)\
                    .to(torch.float32)
            if self.norm_descs:
                query_descs = F.normalize(query_descs)
            residuals = ein.rearrange(query_descs, "q d -> q 1 d") \
                    - ein.rearrange(self.c_centers, "c d -> 1 c d")
            if cache_id is not None and self.can_use_cache_vlad():
                cid_dir = f"{self.cache_dir}/"\
                        f"{os.path.split(cache_id)[0]}"
                if not os.path.isdir(cid_dir):
                    os.makedirs(cid_dir)
                    print(f"Created directory: {cid_dir}")
                torch.save(residuals, 
                        f"{self.cache_dir}/{cache_id}_r.pt")
        # print("residuals",residuals.shape)
        return residuals

    def generate_multi_res_vec(self, 
            multi_query: Union[np.ndarray, torch.Tensor, list],
            cache_ids: Union[List[str], None]=None) \
            -> Union[torch.Tensor, list]:
        """
            Given query descriptors from multiple images, generate
            the VLAD for them.
            
            Parameters:
            - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                            There are 'n_imgs' and each image has
                            'n_kpts' keypoints, with 'd' dimensional
                            descriptor each. If a List (can then have
                            different number of keypoints in each 
                            image), then the result is also a list.
            - cache_ids:    Cache IDs for the VLAD vectors. If None,
                            then no caching is done (stored or 
                            retrieved). If a list, then the length
                            should be 'n_imgs' (one per image).
                            
            Returns:
            - multi_res:    VLAD descriptors for the queries
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate_res_vec(q, c) \
                for (q, c) in zip(multi_query, cache_ids)]
        try:    # Most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:    # Otherwise numpy
                res = np.stack(res)
            except TypeError:
                pass    # Let it remain as a list
        return res

# %%
seed_everything()