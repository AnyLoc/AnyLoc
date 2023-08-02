import os
import sys
from pathlib import Path
# Set the './../' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
    print(dir_name)
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
lib_path = os.path.join(lib_path,'segment-anything')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')

from segment_anything import build_sam, SamPredictor, sam_model_registry
import cv2

sam = sam_model_registry["vit_b"](checkpoint="/ocean/projects/cis220039p/jkarhade/segment-anything/sam_vit_b_01ec64.pth")

img_path = '/ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg/baidu_datasets/query_images_undistort/cdm_20150523_101338.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print('here')
predictor = SamPredictor(sam,use_neck=True,out_layer_num=6)
predictor.set_image(img)
desc = predictor.get_image_embedding()

print('here2')
