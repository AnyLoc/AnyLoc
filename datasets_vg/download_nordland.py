"""
This script download the Nordland dataset with the split used in Patch-NetVLAD.
The images are arranged to be compatible with our benchmarking framework, and
also with the CosPlace repository.
In Nordland usually a prediction for a given query is considered correct if it
is within 10 frames from the query (there's a 1-to-1 match between queries and
database images). Given that our codebases rely on UTM coordinates to find
positives and negatives, we create dummy UTM coordinates for all images,
ensuring that images that are within 10 frames are also within 25 meters.
In practice, we organize all images with UTM_east = 0 (i.e. in a straight line)
and set UTM_north to be 2.4 meters apart between consecutive frames.
"""

import os
import shutil
from tqdm import tqdm
from glob import glob
from PIL import Image
from os.path import join

import util

THRESHOLD_METERS = 25
THRESHOLD_FRAMES = 10
DISTANCE_BETWEEN_FRAMES = THRESHOLD_METERS / (THRESHOLD_FRAMES + 0.5)

datasets_folder = join(os.curdir, "datasets")
dataset_name = "nordland"
dataset_folder = join(datasets_folder, dataset_name)
raw_data_folder = join(datasets_folder, dataset_name, "raw_data")
database_folder = join(dataset_folder, "images", "test", "database")
queries_folder = join(dataset_folder, "images", "test", "queries")
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(database_folder, exist_ok=True)
os.makedirs(queries_folder, exist_ok=True)
os.makedirs(raw_data_folder, exist_ok=True)

util.download_heavy_file("https://cloudstor.aarnet.edu.au/plus/s/8L7loyTZjK0FsWT/download?path=%2F&files=summer.tar.gz",
                          join(raw_data_folder, "summer.tar.gz"))
util.download_heavy_file("https://cloudstor.aarnet.edu.au/plus/s/8L7loyTZjK0FsWT/download?path=%2F&files=winter.tar.gz",
                          join(raw_data_folder, "winter.tar.gz"))
util.download_heavy_file("https://cloudstor.aarnet.edu.au/plus/s/8L7loyTZjK0FsWT/download?path=%2F&files=cleanImageNames.txt&downloadStartSecret=crd03ou9qji",
                          join(raw_data_folder, "cleanImageNames.txt"))

shutil.unpack_archive(join(raw_data_folder, "summer.tar.gz"), raw_data_folder)
shutil.unpack_archive(join(raw_data_folder, "winter.tar.gz"), raw_data_folder)

with open(join(raw_data_folder, "cleanImageNames.txt")) as file:
    selected_images = file.readlines()
    selected_images = [i.replace("\n", "") for i in selected_images]
    selected_images = set(selected_images)

database_paths = sorted(glob(join(raw_data_folder, "summer", "*.png")))
queries_paths = sorted(glob(join(raw_data_folder, "winter", "*.png")))

num_image = 0
for path in tqdm(database_paths, ncols=100):
    if os.path.basename(path) not in selected_images:
        continue
    utm_north = util.format_coord(num_image*DISTANCE_BETWEEN_FRAMES, 5, 1)
    filename = f"@0@{utm_north}@@@@@{num_image}@@@@@@@@.jpg"
    new_path = join(database_folder, filename)
    Image.open(path).save(new_path)
    num_image += 1

num_image = 0
for path in tqdm(queries_paths, ncols=100):
    if os.path.basename(path) not in selected_images:
        continue
    utm_north = util.format_coord(num_image*DISTANCE_BETWEEN_FRAMES, 5, 1)
    filename = f"@0@{utm_north}@@@@@{num_image}@@@@@@@@.jpg"
    new_path = join(queries_folder, filename)
    Image.open(path).save(new_path)
    num_image += 1

