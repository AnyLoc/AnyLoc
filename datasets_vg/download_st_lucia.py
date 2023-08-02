
import os
import shutil
from tqdm import tqdm
from mega import Mega
from skimage import io
from os.path import join

import util
import map_builder

THRESHOLD_IN_METERS = 5

datasets_folder = join(os.curdir, "datasets")
dataset_name = "st_lucia"
dataset_folder = join(datasets_folder, dataset_name)
raw_data_folder = join(datasets_folder, dataset_name, "raw_data")
dst_database_folder = join(dataset_folder, "images", "test", "database")
dst_queries_folder = join(dataset_folder, "images", "test", "queries")
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(dst_database_folder, exist_ok=True)
os.makedirs(dst_queries_folder, exist_ok=True)
os.makedirs(raw_data_folder, exist_ok=True)

# Use the first pass for the database, and the last one for the queries
urls = ['https://mega.nz/file/nE4g0LzZ#c8eL_H3ZfXElqEukw38i32p5cjwusTuNJYYeEP1d5Pg',
        # 'https://mega.nz/file/TVRXlDZR#WUJad1yQunPLpA38Z0rPqBeXXh4g_jnt4n-ZjDF8hKw',
        # 'https://mega.nz/file/LNRkiZAB#LemKJHU7kDunl_9CMQM6xCUdLzVhMYqE3DIZ6QVDOUo',
        # 'https://mega.nz/file/PRJy2BpI#TWSaNioftbPxw7T4LFV3CTtVNN08XnI94WH6_SZEJHE',
        # 'https://mega.nz/file/OFZngZwJ#L3rF3wU69v0Fdfh-iUcKLzkdrUhD6DI6SSYoRZuJkHE',
        # 'https://mega.nz/file/SRJ2wLYQ#EEBH4wCy5je0lbsBNoX9EmGuonYJ2hMothK5w0ehh9U',
        # 'https://mega.nz/file/3U4XnQqT#WifPXhWVmkYw5xAUFyIbaHzjEcghcUfjCK7rqBVl-YY',
        # 'https://mega.nz/file/WN4jzA7Q#Qlg1DxSuYvylIWuZRGFqwjnL1lYV698rsf_eGE1QWnI',
        # 'https://mega.nz/file/fQBmELwa#HHKNU28SreKnQjxIIZ_qUQorWqxZjCraiPT1oiYR4wo',
        'https://mega.nz/file/PAgWSIhD#UeeA6knWL3pDh_IczbYkcA1R1MwSZ2vhEg2DTr1_oNw']
login = Mega().login()

for sequence_num, url in enumerate(urls):
    print(f"{sequence_num:>2} / {len(urls)} ) downloading {url}")
    zip_path = login.download_url(url, raw_data_folder)
    zip_path = str(zip_path)
    subset_name = os.path.basename(zip_path).replace(".zip", "")
    shutil.unpack_archive(zip_path, raw_data_folder)
    
    vr = util.VideoReader(join(raw_data_folder, subset_name, "webcam_video.avi"))
    
    with open(join(raw_data_folder, subset_name, "fGPS.txt"), "r") as file:
        lines = file.readlines()
    
    last_coordinates = None
    for frame_num, line in zip(tqdm(range(vr.frames_num)), lines):
        latitude, longitude = line.split(",")
        latitude = "-" + latitude  # Given latitude is positive, real latitude is negative (in Australia)
        easting, northing = util.format_location_info(latitude, longitude)[:2]
        if last_coordinates is None:
            last_coordinates = (easting, northing)
        else:
            distance_in_meters = util.get_distance(last_coordinates, (easting, northing))
            if distance_in_meters < THRESHOLD_IN_METERS:
                continue  # If this frame is too close to the previous one, skip it
            else:
                last_coordinates = (easting, northing)
        
        frame = vr.get_frame_at_frame_num(frame_num)
        image_name = util.get_dst_image_name(latitude, longitude, pano_id=f"{subset_name}_{frame_num:05d}")
        if sequence_num == 0:  # The first sequence is the database
            io.imsave(join(dst_database_folder, image_name), frame)
        else:
            io.imsave(join(dst_queries_folder, image_name), frame)

map_builder.build_map_from_dataset(dataset_folder)
shutil.rmtree(raw_data_folder)

