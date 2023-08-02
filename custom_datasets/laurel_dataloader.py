# Laurel Caverns dataset
"""
    
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
from typing import List, Union
from natsort import natsorted
from configs import prog_args
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean

from torch.utils.data import DataLoader

import os
import torch
import faiss
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors

from utilities import CustomDataset

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

class Laurel(CustomDataset):
    """
    Returns dataset class with images from database and queries for the gardens dataset. 
    """
    def __init__(self,args,datasets_folder='/home/jay/Downloads/vl_vpr_datasets',dataset_name="laurel_caverns",split="train",use_ang_positives=False,dist_thresh = 8,ang_thresh=20,use_mixVPR=False,use_SAM=False):
        super().__init__()

        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.split = split
        self.use_mixVPR = use_mixVPR
        self.use_SAM = use_SAM
        self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"db_images")))
        self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"q_images")))

        self.db_abs_paths = []
        self.q_abs_paths = []

        for p in self.db_paths:
            self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"db_images",p))

        for q in self.q_paths:
            self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"q_images",q))

        self.db_num = len(self.db_abs_paths)
        self.q_num = len(self.q_abs_paths)
        
        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.gt_positives = np.load(os.path.join(self.datasets_folder,self.dataset_name,"pose_topic_list.npy"),allow_pickle=True) #returns dictionary of gardens dataset
        
        self.db_gt_arr = self.gt_positives[:141,:2]
        self.q_gt_arr = self.gt_positives[229:341,:2]

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_gt_arr)
        self.dist_thresh = dist_thresh
        self.dist,self.soft_positives_per_query = knn.radius_neighbors(self.q_gt_arr,
                                                            radius=self.dist_thresh,
                                                            return_distance=True)            


        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)


    def __getitem__(self, index):
        img = Image.open(self.images_paths[index])

        if self.use_mixVPR:
            img = mixVPR_transform(img)
        elif self.use_SAM:
            img = cv2.imread(self.images_paths[index])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            img = base_transform(img)

        return img, index

if __name__=="__main__":
    args = 0
    vpr_ds = Laurel(args,"/ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg","laurel_caverns",'test')
    print(len(vpr_ds),(vpr_ds.db_num),vpr_ds.q_num)

    print(vpr_ds.q_paths[6], vpr_ds.soft_positives_per_query[6])

