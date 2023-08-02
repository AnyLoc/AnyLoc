"""
A script to trivially perform VPR using off-the-shelf (image-level) CLIP features
"""

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import fast_pytorch_kmeans as fpk
import numpy as np
import open_clip
import torch
import tyro
from natsort import natsorted
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm, trange


@dataclass
class ProgramArgs:

    # Input dir to read images from
    img_dir: Union[Path, str] = "/home/jay/Downloads/data/Berlin/Query"
    # Directory to extract and save features to
    feat_dir: Union[Path, str] = "/home/jay/Downloads/data/Berlin/global_clip"

    # Number of images to skip between successive sampled images
    # e.g., stride = 10 will take every 10th image from `img_dir`
    stride: int = 1

    # Image dimensions (input images will be resized to this before passing to CLIP)
    img_height: int = 120
    img_width: int = 160

    # Directory to save clustered images to
    save_dir: Union[Path, str] = "/home/jay/Downloads/data/Berlin/saved_clustered_sequences"

    # Params of OpenCLIP models. For a complete list of available models, run
    # `openclip.list_pretrained()`
    # OpenCLIP model ID
    clip_model = "ViT-H-14"
    # OpenCLIP model dataset tag
    clip_dataset = "laion2b_s32b_b79k"

    # Whether or not to use PCA to reduce feature dimensionality
    use_pca: bool = False
    n_components: int = 256


if __name__ == "__main__":

    args = tyro.cli(ProgramArgs)

    torch.autograd.set_grad_enabled(False)

    print("Preparing image paths...")
    imgfiles = natsorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))
    imgfiles = imgfiles[:: args.stride]

    os.makedirs(args.feat_dir, exist_ok=True)

    if os.path.exists(args.feat_dir):
        # First, delete all files in the temp dir where we will house our features
        for _file in os.scandir(args.feat_dir):
            os.remove(_file.path)

    print("Initializing CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, args.clip_dataset
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    with torch.cuda.amp.autocast():
        print("Extracting image features...")
        for imgidx in trange(len(imgfiles)):
            stem = os.path.splitext(os.path.basename(imgfiles[imgidx]))[0]
            img = preprocess(Image.open(imgfiles[imgidx])).unsqueeze(0)
            imgfeat = model.encode_image(img)
            imgfeat /= imgfeat.norm(dim=-1, keepdim=True)
            savefile = os.path.join(args.feat_dir, stem + ".pt")
            if imgidx == 1:
                tqdm.write(f"Image feature dims: {imgfeat.shape} \n")
                tqdm.write(f"Saving to {savefile} \n")
            torch.save(imgfeat.detach().cpu(), savefile)

    """
    K-means clustering
    """

    with torch.cuda.amp.autocast():
        print("Loading features...")

        # Only loads features corresponding to images that are read (i.e., takes args.stride into account)
        imgfeat = []
        for imgfile in imgfiles:
            stem = os.path.splitext(os.path.basename(imgfile))[0]
            _featfile = os.path.join(args.feat_dir, stem + ".pt")
            imgfeat.append(torch.load(_featfile))
        imgfeat = torch.cat(imgfeat, dim=0)
        imgfeat = imgfeat.cuda()

        """
        PCA
        """

        if args.use_pca:
            print("Running PCA...")
            pca = PCA(n_components=args.n_components)
            feat_tf = pca.fit_transform(imgfeat.detach().cpu().numpy())
            imgfeat = torch.from_numpy(feat_tf).cuda().float()

        K = 10
        kmeans = fpk.KMeans(n_clusters=K, mode="cosine", verbose=1)
        clusters = kmeans.fit_predict(imgfeat)

        print("Saving clustered images...")
        os.makedirs(args.save_dir, exist_ok=True)
        for _k in range(K):
            os.makedirs(os.path.join(args.save_dir, f"{_k}"), exist_ok=True)
        for imgidx, imgfile in tqdm(enumerate(imgfiles)):
            cluster_idx = clusters[imgidx]
            stem = os.path.splitext(os.path.basename(imgfile))[0]
            savefile = os.path.join(
                args.save_dir, f"{cluster_idx.item()}", stem + ".png"
            )
            img = cv2.imread(imgfile, cv2.IMREAD_ANYCOLOR)
            cv2.imwrite(savefile, img)
