
import os
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from os.path import join
from datetime import datetime

import util
import map_builder

datasets_folder = join(os.curdir, "datasets")
dataset_name = "eynsham"
dataset_folder = join(datasets_folder, dataset_name)
raw_data_folder = join(datasets_folder, dataset_name, "raw_data")
database_folder = join(dataset_folder, "images", "test", "database")
queries_folder = join(dataset_folder, "images", "test", "queries")
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(database_folder, exist_ok=True)
os.makedirs(queries_folder, exist_ok=True)
os.makedirs(raw_data_folder, exist_ok=True)

util.download_heavy_file("https://zenodo.org/record/1243106/files/Eynsham.zip?download=1",
                         join(raw_data_folder, "Eynsham.zip"))

shutil.unpack_archive(join(raw_data_folder, "Eynsham.zip"), raw_data_folder)

with open(join(raw_data_folder, "Eynsham", "Route_map", "Eynsham.kml"), "r") as file:
    lines = file.readlines()

lines = [l.replace("\n", "") for l in lines]
text = lines[11]
splits = text.split("<coordinates>")[1].split("</coordinates>")[0].split(" ")[:-2]
coords = np.array([s.split(",")[:2] for s in splits]).astype(np.float64)

src_images_paths = sorted(glob(join(raw_data_folder, "Eynsham", "Images", "*.ppm")))[5:]

for pano_num, (lon, lat) in enumerate(tqdm(coords, ncols=100)):
    for tile_num in range(5):
        src_image_path = src_images_paths[pano_num*5 + tile_num]
        timestamp = src_image_path.split("grab_")[1].split(".")[0]
        timestamp = datetime.utcfromtimestamp(int(timestamp)).strftime('%Y%m%d_%H%M%S')
        dst_image_name = util.get_dst_image_name(lat, lon, pano_id=f"{pano_num:04d}",
                                             tile_num=tile_num, timestamp=timestamp)
        # The first 4787 images correspond to the first sequence, and it is the database
        if pano_num < 4787:
            Image.open(src_image_path).save(join(database_folder, dst_image_name))
        else:
            Image.open(src_image_path).save(join(queries_folder, dst_image_name))

map_builder.build_map_from_dataset(dataset_folder)
shutil.rmtree(raw_data_folder)

