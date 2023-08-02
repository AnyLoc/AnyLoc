
import os
import shutil

import gdown

datasets_folder = os.path.join(os.curdir, "datasets")
os.makedirs(datasets_folder, exist_ok=True)
zip_filepath = os.path.join(datasets_folder, 'svox.zip')
gdown.download(id="16iuk8voW65GaywNUQlWAbDt6HZzAJ_t9",
               output=zip_filepath,
               quiet=False)

shutil.unpack_archive(zip_filepath, datasets_folder)
os.remove(zip_filepath)
