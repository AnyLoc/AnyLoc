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

from utilities import CustomDataset

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
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

class Oxford(torch.utils.data.Dataset, CustomDataset):
    def __init__(self,datasets_folder=prog_args.data_vg_dir,dataset_name="Oxford",split="train", override_dist=None, input_transform=True, onlyDB=False,use_mixVPR=False,use_SAM=False):
        super().__init__()

        self.use_mixVPR = use_mixVPR
        self.use_SAM = use_SAM

        structFile = os.path.join(datasets_folder,"Oxford_Robotcar/oxdatapart.mat")
        root_dir = os.path.join(datasets_folder,"Oxford_Robotcar/oxDataPart")

        self.input_transform = input_transform
        
        if self.use_SAM:
            self.input_transform = False
        
        self.dbStruct = parse_dbStruct(structFile)
        if override_dist is not None:   # Override localization radius
            self.loc_rad = override_dist
        else:
            self.loc_rad = self.dbStruct.posDistThr # From file
        self.images = [join(root_dir, dbIm.replace(' ','')) for dbIm in self.dbStruct.dbImage]
            
        if not onlyDB:
            self.images += [join(root_dir, qIm.replace(' ','')) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.db_num = self.dbStruct.numDb
        self.q_num = self.dbStruct.numQ

        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.soft_positives_per_query = None
        self.soft_positives_per_db = None
        self.distances = None

        self._imgs_level = 3    # 3 levels for accessing images
        self.images_paths = self.images # Alias for workings
        self.get_positives()
        self.get_positives_db()

    def get_image_paths(self):
        return self.images

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            if self.use_mixVPR:
                img = mixVPR_transform(img)
            else:
                img = base_transform(img)
        
        if self.use_SAM:
            img = cv2.imread(self.images[index])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        return img, index

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.soft_positives_per_query is None:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.locDb)

            self.distances, self.soft_positives_per_query = knn.radius_neighbors(self.dbStruct.locQ,
                    radius=self.loc_rad)

        return self.soft_positives_per_query

    def get_positives_db(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.soft_positives_per_db is None:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.locDb)

            self.distances, self.soft_positives_per_db = knn.radius_neighbors(self.dbStruct.locDb,
                    radius=self.loc_rad)

        return self.soft_positives_per_db
