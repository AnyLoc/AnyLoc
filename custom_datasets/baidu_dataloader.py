# Baidu Mall dataset
"""
    Download from: https://www.dropbox.com/s/4mksiwkxb7t4a8a/IDL_dataset_cvpr17_3852.zip
    jar xf ./IDL_dataset_cvpr17_3852.zip
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
from typing import Union, List

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

def get_cop_pose(file):
    """
    Takes in input of .camera file for baidu and outputs the cop numpy array [x y z] and 3x3 rotation matrix
    """
    with open(file) as f:
        lines = f.readlines()
        xyz_cop_line = lines[-2]
        # print(cop_line)
        xyz_cop = np.fromstring(xyz_cop_line, dtype=float, sep=' ')    

        r1 = np.fromstring(lines[4], dtype=float, sep=' ')
        r2 = np.fromstring(lines[5], dtype=float, sep=' ')
        r3 = np.fromstring(lines[6], dtype=float, sep=' ')
        r =  Rotation.from_matrix(np.array([r1,r2,r3]))
        # print(R)

        R_euler = r.as_euler('zyx', degrees=True)

    return xyz_cop,R_euler


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

class Baidu_Dataset(torch.utils.data.Dataset, CustomDataset):
    """
    Return dataset class with images from database and queries for the Baidu dataset
    """

    def __init__(self,args,datasets_folder=prog_args.data_vg_dir,dataset_name="baidu_datasets",split="train",use_ang_positives=False,dist_thresh = 10,ang_thresh=20,use_mixVPR=False,use_SAM=False):
        super().__init__()

        self.use_mixVPR = use_mixVPR
        self.use_SAM = use_SAM

        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.split = split

        self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"training_images_undistort")))
        self.db_gt_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"training_gt")))
        self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"query_images_undistort")))
        self.q_gt_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"query_gt")))

        self.db_abs_paths = []
        self.q_abs_paths = []

        for p in self.db_paths:
            self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"training_images_undistort",p))

        for q in self.q_paths:
            self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"query_images_undistort",q))

        self.ang_thresh = ang_thresh
        self.dist_thresh = dist_thresh

        self.resize = args.resize
        self.test_method = args.test_method

        self.db_num = len(self.db_paths)
        self.q_num = len(self.q_paths)
        
        self.database_num = self.db_num
        self.queries_num = self.q_num

        #form pose array from db_gt .camera files
        self.db_gt_arr = np.zeros((self.db_num,3)) #for xyz
        self.db_gt_arr_euler = np.zeros((self.db_num,3)) #for euler angles

        for idx,db_gt_file_rel in enumerate(self.db_gt_paths):

            db_gt_file = os.path.join(self.datasets_folder,self.dataset_name,"training_gt",db_gt_file_rel)

            with open(db_gt_file) as f:
                cop_pose,cop_R = get_cop_pose(db_gt_file)
                
            self.db_gt_arr[idx,:] = cop_pose
            self.db_gt_arr_euler[idx,:] = cop_R

        #form pose array from q_gt .camera files
        self.q_gt_arr = np.zeros((self.q_num,3)) #for xyz
        self.q_gt_arr_euler = np.zeros((self.q_num,3)) #for euler angles

        for idx,q_gt_file_rel in enumerate(self.q_gt_paths):

            q_gt_file = os.path.join(self.datasets_folder,self.dataset_name,"query_gt",q_gt_file_rel)

            with open(q_gt_file) as f:
                cop_pose,cop_R = get_cop_pose(q_gt_file)
            
            self.q_gt_arr[idx,:] = cop_pose
            self.q_gt_arr_euler[idx,:] = cop_R

        if use_ang_positives:
            # Find soft_positives_per_query, which are within val_positive_dist_threshold and ang_threshold
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist,self.soft_dist_positives_per_query = knn.radius_neighbors(self.q_gt_arr,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)

            #also apply the angular distance threshold
            self.soft_positives_per_query = []

            for i in range(len(self.q_gt_arr)):                 #iterate over all q_gt_array
                self.ang_dist = []
                for j in range(len(self.soft_dist_positives_per_query[i])): #iterate over all positive queries
                    # print(self.q_gt_arr - self.db_gt_arr[self.soft_positives_per_query[i][j]])
                    ang_diff = np.mean(np.abs(self.q_gt_arr_euler[i] - self.db_gt_arr_euler[self.soft_dist_positives_per_query[i][j]]))
                    if ang_diff<self.ang_thresh:
                        self.ang_dist.append(self.soft_dist_positives_per_query[i][j])
                self.soft_positives_per_query.append(self.ang_dist)

            #Shallow MLP Training Database
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist,self.soft_dist_positives_per_db = knn.radius_neighbors(self.db_gt_arr,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)

            self.soft_positives_per_db = []

            for i in range(len(self.db_gt_arr)):                 #iterate over all q_gt_array
                self.ang_dist = []
                for j in range(len(self.soft_dist_positives_per_db[i])): #iterate over all positive queries
                    # print(self.q_gt_arr - self.db_gt_arr[self.soft_positives_per_db[i][j]])
                    ang_diff = np.mean(np.abs(self.q_gt_arr_euler[i] - self.db_gt_arr_euler[self.soft_dist_positives_per_db[i][j]]))
                    if ang_diff<self.ang_thresh:
                        self.ang_dist.append(self.soft_dist_positives_per_db[i][j])
                self.soft_positives_per_db.append(self.ang_dist)

        else :
            # Find soft_positives_per_query, which are within val_positive_dist_threshold only
            # self.db_gt_arr = self.db_gt_arr.tolist()
            # self.q_gt_arr = self.q_gt_arr.tolist()
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist,self.soft_positives_per_query = knn.radius_neighbors(self.q_gt_arr,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)            


            #Shallow MLP Training for database
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist,self.soft_positives_per_db = knn.radius_neighbors(self.db_gt_arr,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)            


        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

    def __getitem__(self,index):
        img = path_to_pil_img(self.images_paths[index])
        # if self.split=="train":
        #     img = path_to_pil_img(self.db_paths[index])
        # elif self.split=="test":
        #     img = path_to_pil_img(self.q_paths[index])
        
        if self.use_mixVPR:
            img = mixVPR_transform(img)
        elif self.use_SAM:
            img = cv2.imread(self.images_paths[index])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            img = base_transform(img)
            # With database images self.test_method should always be "hard_resize"
            if self.test_method == "hard_resize":
                # self.test_method=="hard_resize" is the default, resizes all images to the same size.
                img = T.functional.resize(img, self.resize)
            else:
                img = self._test_query_transform(img)
        return img, index

    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = T.functional.resize(img, min(self.resize))
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = T.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = T.functional.resize(img, shorter_side)
            processed_img = torch.stack(T.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img


if __name__=="__main__":

    vpr_ds = Baidu_Dataset()
    vpr_dl = DataLoader(vpr_ds, 1, pin_memory=True, shuffle=False)
    # print(vpr_ds.soft_positives_per_query[0])

    q_idx = 10
    print(vpr_ds.q_paths[q_idx])
    print(len(vpr_ds.soft_positives_per_query[q_idx]))#,len(vpr_ds.soft_dist_positives_per_query[q_idx]))
    for i in range(len(vpr_ds.soft_positives_per_query[q_idx])):
        print(vpr_ds.db_paths[vpr_ds.soft_positives_per_query[q_idx][i]],vpr_ds.dist[q_idx][i])


    # db_descs, qu_descs, pos_pq = build_cache(largs, vpr_dl, model)
