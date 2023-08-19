# Generate VLAD global descriptors using AnyLoc-DINOv2
"""
    Ensure that this repository is cloned.
    
    We'll use the CityCenter dataset as an example [1]. The dataset
    will be downloaded automatically if `use_example` is set to True.
    Don't change the defaults (except `out_dir`) for the example 
    dataset.
    
    To use a custom dataset, set `use_example` to False and ensure
    that the input directory contains the images. The images should
    be in a single folder and the folder should contain only the
    images.
    
    [1]: https://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/
"""

# %%
import torch
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from PIL import Image
import numpy as np
import tyro
import os
import sys
import glob
import natsort
from tqdm.auto import tqdm
from dataclasses import dataclass
from onedrivedownloader import download
from utilities import od_down_links
from typing import Literal, Union
# DINOv2 imports
from utilities import DinoV2ExtractFeatures
from utilities import VLAD


# %%
@dataclass
class LocalArgs:
    # If True, the example dataset will be downloaded
    use_example: bool = True
    """
        If True, the example dataset will be downloaded. If False,
        then the input directory should contain the images (user is
        expected to arrange it).
    """
    # Input directory containing images
    in_dir: str = "./data/CityCenter/Images"
    # Image file extension
    imgs_ext: str = "jpg"
    # Output directory where global descriptors will be stored
    out_dir: str = "./data/CityCenter/GD_Images"
    # Maximum edge length (expected) across all images (GPU OOM)
    max_img_size: int = 1024
    # Use the OneDrive mirror for example
    use_od_example: bool = True
    # Use only the first-N images (for testing). Use all if 'None'.
    first_n: Union[int, None] = None
    # Domain to use for loading VLAD cluster centers
    domain: Literal["aerial", "indoor", "urban"] = "urban"
    # Number of clusters (cluster centers for VLAD) - read from cache
    num_c: int = 32


# %%
# Download the cache (if doesn't exist)
def download_cache():
    l = od_down_links["cache"]  # Link
    if os.path.isdir("./cache"):
        print("Cache folder already exists!")
    else:
        print("Downloading the cache folder")
        download(l, filename="cache.zip", unzip=True, unzip_path="./")
        print("Cache folder downloaded")


# Download the testing dataset
def download_test_data(use_odrive:bool):
    if use_odrive:
        print("Downloading images from OneDrive ...")
        imgs_link = od_down_links["test_imgs_od"]
        download(imgs_link, "./data/CityCenter/Images.zip", unzip=True, unzip_path="./data/CityCenter")
        print("Download and extraction of images from OneDrive completed")
    else:
        print("Downloading from original source")
        imgs_link = od_down_links["test_imgs"]
        if os.path.isdir("./data/CityCenter"):
            print("Directory already exists")
        else:
            os.makedirs("./data/CityCenter")
            print("Directory created")
        os.system(f"wget {imgs_link} -O ./data/CityCenter/Images.zip")
        print("Extraction completed")
    print("Dataset is ready to test")


# %%
def main(largs: LocalArgs):
    # Basic utilities
    _ex = lambda x: os.path.realpath(os.path.expanduser(x))
    
    # Ensure that cache and data is there
    download_cache()
    if largs.use_example:
        print("Using the example dataset")
        download_test_data(use_odrive=largs.use_od_example)
    else:
        print("Using the custom dataset")
    
    # Program parameters
    save_dir = _ex(largs.out_dir)
    device = torch.device("cuda")
    # Dino_v2 properties (parameters)
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    num_c: int = largs.num_c
    # Domain for use case (deployment environment)
    domain: largs.domain
    # Maximum image dimension
    max_img_size: int = largs.max_img_size
    # Ensure inputs are fine
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print(f"Creating directory: {save_dir}")
    else:
        print("Save directory already exists, overwriting possible!")
    
    # Load the DINO extractor model
    extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
        desc_facet, device=device)
    base_tf = tvf.Compose([ # Base image transformations
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])
    
    # VLAD object (load cache)
    cache_dir = _ex("./cache")
    ext_specifier = f"dinov2_vitg14/"\
            f"l{desc_layer}_{desc_facet}_c{num_c}"
    c_centers_file = os.path.join(cache_dir, "vocabulary", 
            ext_specifier, domain, "c_centers.pt")
    assert os.path.isfile(c_centers_file), "Vocabulary not cached!"
    c_centers = torch.load(c_centers_file)
    assert c_centers.shape[0] == num_c, "Wrong number of clusters!"
    # Main VLAD object
    vlad = VLAD(num_c, desc_dim=None, 
        cache_dir=os.path.dirname(c_centers_file))
    vlad.fit(None)  # Load the vocabulary
    
    # Global descriptor generation
    imgs_dir = _ex(largs.in_dir)
    assert os.path.isdir(imgs_dir), "Input directory doesn't exist!"
    img_fnames = glob.glob(f"{imgs_dir}/*.jpg")
    img_fnames = natsort.natsorted(img_fnames)
    if largs.first_n is not None:
        img_fnames = img_fnames[:largs.first_n]
    for img_fname in tqdm(img_fnames):
        # DINO features
        with torch.no_grad():
            pil_img = Image.open(img_fname).convert('RGB')
            img_pt = base_tf(pil_img).to(device)
            if max(img_pt.shape[-2:]) > max_img_size:
                c, h, w = img_pt.shape
                # Maintain aspect ratio
                if h == max(img_pt.shape[-2:]):
                    w = int(w * max_img_size / h)
                    h = max_img_size
                else:
                    h = int(h * max_img_size / w)
                    w = max_img_size
                print(f"To {(h, w) =}")
                img_pt = T.resize(img_pt, (h, w), 
                        interpolation=T.InterpolationMode.BICUBIC)
                print(f"Resized {img_fname} to {img_pt.shape = }")
            # Make image patchable (14, 14 patches)
            c, h, w = img_pt.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
            # Extract descriptor
            ret = extractor(img_pt) # [1, num_patches, desc_dim]
        # VLAD global descriptor
        gd = vlad.generate(ret.cpu().squeeze()) # VLAD:  [agg_dim]
        gd_np = gd.numpy()[np.newaxis, ...] # shape: [1, agg_dim]
        np.save(f"{save_dir}/{os.path.basename(img_fname)}.npy",
                gd_np)


# %%
if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    args = tyro.cli(LocalArgs, description=__doc__)
    main(args)
    print("Exiting program")
    exit(0)


# %%
