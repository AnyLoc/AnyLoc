
import os
import shutil
from glob import glob
from tqdm import tqdm
from os.path import join

import util

# This dictionary is copied from the original code
# https://github.com/mapillary/mapillary_sls/blob/master/mapillary_sls/datasets/msls.py#L16
default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam","helsinki",
              "tokyo","toronto","saopaulo","moscow","zurich","paris","bangkok",
              "budapest","austin","berlin","ottawa","phoenix","goa","amman","nairobi","manila"],
    'val': ["cph", "sf"],
    'test': ["miami","athens","buenosaires","stockholm","bengaluru","kampala"]
}

csv_files_paths = sorted(glob(join("datasets", "mapillary_sls", "*", "*", "*", "postprocessed.csv"),
                              recursive=True))

for csv_file_path in csv_files_paths:
    with open(csv_file_path, "r") as file:
        postprocessed_lines = file.readlines()[1:]
    with open(csv_file_path.replace("postprocessed", "raw"), "r") as file:
        raw_lines = file.readlines()[1:]
    assert len(raw_lines) == len(postprocessed_lines)

    csv_dir = os.path.dirname(csv_file_path)
    city_path, folder = os.path.split(csv_dir)
    city = os.path.split(city_path)[1]

    folder = "database" if folder == "database" else "queries"
    train_val = "train" if city in default_cities["train"] else "val"
    dst_folder = os.path.join('msls', train_val, folder)

    os.makedirs(dst_folder, exist_ok=True)
    for postprocessed_line, raw_line in zip(tqdm(postprocessed_lines, desc=city), raw_lines):
        _, pano_id, lon, lat, _, timestamp, is_panorama = raw_line.split(",")
        if is_panorama == "True\n":
            continue
        timestamp = timestamp.replace("-", "")
        view_direction = postprocessed_line.split(",")[-1].replace("\n", "").lower()
        day_night = "day" if postprocessed_line.split(",")[-2] == "False" else "night"
        note = f"{day_night}_{view_direction}_{city}"
        dst_image_name = util.get_dst_image_name(lat, lon, pano_id, timestamp=timestamp, note=note)
        src_image_path = os.path.join(os.path.dirname(csv_file_path), 'images', f'{pano_id}.jpg')
        dst_image_path = os.path.join(dst_folder, dst_image_name)
        _ = shutil.move(src_image_path, dst_image_path)

val_path = os.path.join('msls', 'val')
test_path = os.path.join('msls', 'test')
os.symlink(os.path.abspath(val_path), test_path)
