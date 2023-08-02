# Eiffel point dataset
"""
    From: Custom Assembled Dataset from 2015 and 2020 runs on Eiffel Tower
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

def get_poses(pose_path):
  # Read and parse the poses
  poses = []
  try:
    if '.txt' in pose_path:
      with open(pose_path, 'r') as f:
        lines = f.readlines()#[::2]
        for line in lines:
          l_split = line.split(' ')
          #print(l_split)
          pose = [float(l_split[1]),float(l_split[2]),float(l_split[3])]
          #print(poses)
          poses.append(pose)
    else:
      poses = np.load(pose_path)['arr_0']
  
  except FileNotFoundError:
    print('Ground truth poses are not avaialble.')
  
  return np.array(poses)
    

base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Resize((405,720)) # Prevent OOM on GPU (with same aspect ratio)
])

mixVPR_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    T.Resize((320,320))
])

class Eiffel(CustomDataset):
    """
    Returns dataset class with images from database and queries for the gardens dataset. 
    """
    def __init__(self,args,datasets_folder='/home/jay/Downloads/vl_vpr_datasets',dataset_name="eiffel",split="train",use_ang_positives=False,dist_thresh = 1,ang_thresh=20,use_mixVPR=False,use_SAM=False):
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

        self.gt_positives = np.load(os.path.join(self.datasets_folder,self.dataset_name,"eiffel_gt.npy"),allow_pickle=True)[101:] #returns dictionary of gardens dataset

        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

        self.soft_positives_per_query = []

        for i in range(len(self.gt_positives)):
            self.soft_positives_per_query.append(self.gt_positives[i][1]) #corresponds to index wise soft positives


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
    vpr_ds = Eiffel(args,"/ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg","eiffel",'test')
    print(len(vpr_ds),(vpr_ds.db_num),vpr_ds.q_num)

    print(vpr_ds.q_paths[6], vpr_ds.soft_positives_per_query[6])

