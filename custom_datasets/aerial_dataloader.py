# Tartan GNSS dataset
"""
    Has two versions: `rotated` and `notrotated`
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
import pandas as pd
import os
import numpy as np
import cv2
import torch
import torch.utils.data 
from natsort import natsorted
from configs import prog_args
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean
from typing import List, Union
from torch.utils.data import DataLoader

import os
import torch
import faiss
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mixVPR_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    T.Resize((320,320))
])

class Aerial():
    """
    Returns dataset class with images from database and queries for the gardens dataset. 
    """
    def __init__(self,args,datasets_folder='/home/jay/Downloads/vl_vpr_datasets',dataset_name="Tartan_GNSS_rotated",split="train",use_mixVPR=False,use_SAM=False):
        super().__init__()

        if dataset_name == "Tartan_GNSS_rotated":
            self.dataset_name = "gnss_train_rotated"
        elif dataset_name == "Tartan_GNSS_notrotated":
            self.dataset_name = "gnss_train_notrotated"
        elif dataset_name == "Tartan_GNSS_test_notrotated":
            self.dataset_name = "test_40_midref_rot0"
        elif dataset_name == "Tartan_GNSS_test_rotated":
            self.dataset_name = "test_40_midref_rot90"
        else:
            raise NotImplementedError(f"Dataset: {dataset_name}")
        self.datasets_folder = datasets_folder
        self.split = split
        self.use_mixVPR = use_mixVPR
        self.use_SAM = use_SAM

        self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"reference_images")))
        self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"query_images")))

        self.db_abs_paths = []
        self.q_abs_paths = []

        for p in self.db_paths:
            self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"reference_images",p))

        for q in self.q_paths:
            self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"query_images",q))

        self.db_num = len(self.db_abs_paths)
        self.q_num = len(self.q_abs_paths)
        
        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.gt_positives = pd.read_csv(os.path.join(self.datasets_folder,self.dataset_name,"gt_matches.csv"))

        self.db_abs_paths = [os.path.join(self.datasets_folder,self.dataset_name,"reference_images",path) for path in self.db_paths]
        self.q_abs_paths = [os.path.join(self.datasets_folder,self.dataset_name,"query_images",path) for path in self.q_paths]
        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

        # print(f"{self.db_abs_paths[:10]}")

        self.soft_positives_per_query = []

        for i in range(len(self.gt_positives['query_ind'])):
            curr_soft_pos = []
            for top_idx in range(1,6):
                curr_soft_pos.append(self.gt_positives['top_{}_ref_ind'.format(top_idx)][i])

            self.soft_positives_per_query.append(curr_soft_pos)

    def get_image_paths(self):
        return self.images_paths

    def get_image_relpaths(self, i: Union[int, List[int]]) \
            -> Union[List[str], str]:
        """
            Get the relative path of the image at index i. Multiple 
            indices can be passed as a list (or int-like array).
        """
        indices = i
        if type(i) == int:
            indices = [i]
        img_paths = self.get_image_paths()
        s = 3
        rel_paths = ["/".join(img_paths[k].split("/")[-s:]) \
                        for k in indices]
        if type(i) == int:
            return rel_paths[0]
        return rel_paths

    def __getitem__(self, index):
        img = Image.open(self.images_paths[index])
        # img = cv2.imread(self.images_paths[index])
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.use_mixVPR:
            img = mixVPR_transform(img)
        elif self.use_SAM:
            img = cv2.imread(self.images_paths[index])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            img = base_transform(img)

        return img, index

    def __len__(self):
        return len(self.images_paths)

    def get_positives(self):
        """
        Return positives
        """
        return self.soft_positives_per_query
