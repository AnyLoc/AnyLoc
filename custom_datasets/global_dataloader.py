import pandas as pd
import os
import numpy as np
import cv2
import torch
import torch.utils.data 
from natsort import natsorted

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

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
from configs import prog_args
from sklearn.neighbors import NearestNeighbors
import cv2
from typing import List, Union
from glob import glob

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

# base_transform = T.Compose([
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        transforms.Resize((320,320))
    ])

mixVPR_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    transforms.Resize((320,320))
])

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'locDb', 'qImage', 'locQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)

    matStruct = mat['dbStruct'][0]

    dataset = 'dataset'

    whichSet = 'VPR'

    dbImage = matStruct[0]
    locDb = matStruct[1]

    qImage = matStruct[2]
    locQ = matStruct[3]

    numDb = matStruct[4].item()
    numQ = matStruct[5].item()

    posDistThr = matStruct[6].item()
    posDistSqThr = matStruct[7].item()

    return dbStruct(whichSet, dataset, dbImage, locDb, qImage, 
            locQ, numDb, numQ, posDistThr, 
            posDistSqThr)

class Global_Dataloader():
    """
    Returns dataset class with images from database and queries for the gardens dataset. 
    """
    def __init__(self,args,datasets_folder='/home/jay/Downloads/vl_vpr_datasets',dataset_name_list=["gardens"],split="train",use_ang_positives=False,dist_thresh = 10,ang_thresh=20,use_mixVPR=False,use_SAM=False):
        super().__init__()

        self.images_paths = []

        for dataset_name in dataset_name_list:

            self.dataset_name_list = dataset_name_list
            self.datasets_folder = datasets_folder
            self.split = split
            self.dataset_name = dataset_name

            # gardens
            if dataset_name=='gardens':
                
                self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"day_right")))
                self.db_abs_paths = []

                for p in self.db_paths:
                    self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"day_right",p))

                self.images_paths = self.images_paths + list(self.db_abs_paths)#[::3]
                print("number of images in gardens",len(self.images_paths))

            # oxford
            elif dataset_name=='Oxford':
                
                structFile = os.path.join(datasets_folder,"Oxford_Robotcar/oxdatapart.mat")
                root_dir = os.path.join(datasets_folder,"Oxford_Robotcar/oxDataPart")
                
                self.dbStruct = parse_dbStruct(structFile)
                self.images_paths = self.images_paths + [os.path.join(root_dir, dbIm.replace(' ','')) for dbIm in self.dbStruct.dbImage]#[::3]
                print("number of images in oxford",len(self.images_paths))  

            # baidu
            elif dataset_name=="baidu_datasets":
                
                self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"training_images_undistort")))
                self.db_abs_paths = []

                for p in self.db_paths:
                    self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"training_images_undistort",p))

                self.images_paths = self.images_paths + list(self.db_abs_paths)#[::3]
                print("number of images in baidu dataset",len(self.images_paths))
            
            # eiffel
            elif dataset_name=="eiffel":
                self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"db_images")))
                self.db_abs_paths = []

                for p in self.db_paths:
                    self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"db_images",p))

                self.images_paths = self.images_paths + list(self.db_abs_paths)#[::3]
                print("number of images in eiffel dataset",len(self.images_paths))

            #hawkins
            elif dataset_name=="hawkins_long_corridor":
                self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"db_images")))
                self.db_abs_paths = []

                for p in self.db_paths:
                    self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"db_images",p))

                self.images_paths = self.images_paths + list(self.db_abs_paths)#[::3]
                print("number of images in hawkins dataset",len(self.images_paths))

            #laurel caverns
            elif dataset_name=="laurel_caverns":
                self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"db_images")))
                self.db_abs_paths = []

                for p in self.db_paths:
                    self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"db_images",p))

                self.images_paths = self.images_paths + list(self.db_abs_paths)#[::3]
                print("number of images in hawkins dataset",len(self.images_paths))

            #VPAir
            elif dataset_name=="VPAir":
                self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"reference_views")))
                self.db_abs_paths = []

                for p in self.db_paths:
                    self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"reference_views",p))

                self.images_paths = self.images_paths + list(self.db_abs_paths)#[::3]
                print("number of images in VPAir dataset",len(self.images_paths))

            #GNSS Tartan
            elif dataset_name=="GNSS_Tartan":
                self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"reference_images")))
                self.db_abs_paths = []

                for p in self.db_paths:
                    self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"reference_images",p))

                self.images_paths = self.images_paths + list(self.db_abs_paths)#[::3]
                print("number of images in GNSS Tartan dataset",len(self.images_paths))

            # other vg datasets
            else:
                self.args = args
                self.dataset_name = dataset_name
                self.datasets_folder = os.path.join(datasets_folder, dataset_name)
                self.use_mixVPR = use_mixVPR
                self.use_SAM = use_SAM
                self.vprbench = False
                if "ref" in os.listdir(self.datasets_folder):
                    self.vprbench = True
                    database_folder_name, queries_folder_name = "ref", "query"
                else:
                    self.datasets_folder = join(self.datasets_folder, "images", split)
                    database_folder_name = "database"
                if not os.path.exists(self.datasets_folder):
                    raise FileNotFoundError(f"Folder {self.datasets_folder} does not exist")
                
                self.resize = args.resize
                self.test_method = args.test_method
                
                #### Read paths and UTM coordinates for all images.
                database_folder = join(self.datasets_folder, database_folder_name)
                if not os.path.exists(database_folder):
                    raise FileNotFoundError(f"Folder {database_folder} does not exist")
                self.database_paths = natsorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
                
                if dataset_name=="pitts30k":
                    self.images_paths = self.images_paths + list(self.database_paths)[::10]
                elif dataset_name=="st_lucia":
                    self.images_paths = self.images_paths + list(self.database_paths)[::2]
                else:
                    self.images_paths = self.images_paths + list(self.database_paths)#[::1]

                print("number of images in vg dataset",len(self.images_paths))

            print("dataset :",dataset_name)
    
    def __getitem__(self, index):
        img = Image.open(self.images_paths[index])

        img = base_transform(img)

        return img, index

    def __len__(self):
        return len(self.images_paths)

