import pandas as pd
import os
import numpy as np
import cv2
import torch
import torch.utils.data 
from natsort import natsorted

from scipy.spatial.transform import Rotation

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

def get_poses(file_path):
    """
    Get the file and return array of Nx3 (position) and Nx3 (euler angles)
    """
    with open(file_path) as f:
        all_lines = f.readlines()[2:]
        loc_arr = np.zeros((len(all_lines),3))
        euler_arr = np.zeros((len(all_lines),3))
        gt_img_list = []
        # print(loc_arr.shape)
        for idx,l in enumerate(all_lines):
            # print(l.split(","))
            cur_line = l.split(",")
            img_name = l.split(",")[1].split("_")[0] + "_" + l.split(",")[0]
            # print((float(cur_line[2])))
            loc_arr[idx] = np.array([float(cur_line[6]), float(cur_line[7]), float(cur_line[8])])
            quat = [float(cur_line[3]), float(cur_line[4]), float(cur_line[5]), float(cur_line[2])]
            # (print(loc_arr[idx],quat))
            r = Rotation.from_quat(quat)
            R_euler = r.as_euler('zyx', degrees=True)
            # print(R_euler)
            euler_arr[idx] = R_euler
            gt_img_list.append(img_name)

    return gt_img_list,loc_arr,euler_arr


base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class NVL_Dataset(torch.utils.data.Dataset):
    """
    Return dataset class with images from database and queries for the NVL Dataset
    """

    def __init__(self,args,datasets_folder='/home/jay/Downloads/vl_vpr_datasets',dataset_name="NVL_datasets/",split="train",use_soft_positives=True,dist_thresh = 20,ang_thresh=10):
        super().__init__()
        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.split = split

        self.ang_thresh = ang_thresh
        self.dist_thresh = dist_thresh

        self.resize = args.resize
        self.test_method = args.test_method

        self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"database_images")))
        self.db_gt_path = os.path.join(self.datasets_folder,self.dataset_name,"db_trajectories.txt")
        self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"query_images")))
        self.q_gt_path = os.path.join(self.datasets_folder,self.dataset_name,"q_trajectories.txt")

        self.db_num = len(self.db_paths)
        self.q_num = len(self.q_paths)

        self.db_abs_paths = []
        self.q_abs_paths = []

        for p in self.db_paths:
            self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"database_images",p))

        for q in self.q_paths:
            self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"query_images",q))

        #form pose array from db_gt .camera files

        self.db_img_list, self.db_pos_list, self.db_euler_list = get_poses(self.db_gt_path)
        self.q_img_list, self.q_pos_list, self.q_euler_list = get_poses(self.q_gt_path)
        
        if use_soft_positives:
            # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_pos_list)
            self.dist,self.soft_dist_positives_per_query = knn.radius_neighbors(self.q_pos_list,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)

            #also apply the angular distance threshold
            self.soft_positives_per_query = []

            for i in range(len(self.q_pos_list)):                 #iterate over all q_gt_array
                self.ang_dist = []
                for j in range(len(self.soft_dist_positives_per_query[i])): #iterate over all positive queries
                    # print(np.abs(self.q_euler_list[i] - self.db_euler_list[self.soft_dist_positives_per_query[i][j]]))
                    ang_diff = np.mean(np.abs(self.q_euler_list[i] - self.db_euler_list[self.soft_dist_positives_per_query[i][j]]))
                    # print(ang_diff)
                    if ang_diff<self.ang_thresh:
                        self.ang_dist.append(self.soft_dist_positives_per_query[i][j])
                self.soft_positives_per_query.append(self.ang_dist)

        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

    def get_image_paths(self):
        return self.images_paths

    def __getitem__(self,index):
        img = path_to_pil_img(self.images_paths[index])
        # if self.split=="train":
        #     img = path_to_pil_img(self.db_paths[index])
        # elif self.split=="test":
        #     img = path_to_pil_img(self.q_paths[index])
        img = base_transform(img)
        # With database images self.test_method should always be "hard_resize"
        if self.test_method == "hard_resize":
            # self.test_method=="hard_resize" is the default, resizes all images to the same size.
            img = T.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
        return img, index

    def __len__(self):
        return len(self.images_paths)

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

    def get_positives(self):
        """
        Return positives
        """
        return self.soft_positives_per_query

if __name__=="__main__":

    vpr_ds = NVL_Dataset()
    vpr_dl = DataLoader(vpr_ds, 1, pin_memory=True, shuffle=False)

    for q_idx in range(1000):
        print(vpr_ds.q_img_list[q_idx])
        print(len(vpr_ds.soft_positives_per_query[q_idx]),len(vpr_ds.soft_dist_positives_per_query[q_idx]))
        for i in range(len(vpr_ds.soft_positives_per_query[q_idx])):
            print(vpr_ds.db_img_list[vpr_ds.soft_positives_per_query[q_idx][i]],vpr_ds.dist[q_idx][i],vpr_ds.db_euler_list[vpr_ds.soft_positives_per_query[q_idx][i]],vpr_ds.q_euler_list[q_idx])


    # db_descs, qu_descs, pos_pq = build_cache(largs, vpr_dl, model)
