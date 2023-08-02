# Run LSeg work

datasets_vg_dir="/scratch/avneesh.mishra/lseg/datasets_vg_cache"
# Database directories for each dataset
declare -A db_dirs
db_dirs["Oxford"]="$datasets_vg_dir/Oxford_Robotcar/oxDataPart/1-s-resized"
db_dirs["gardens"]="$datasets_vg_dir/gardens/day_right"
db_dirs["st_lucia"]="$datasets_vg_dir/st_lucia/test/database"
db_dirs["17places"]="$datasets_vg_dir/17places/ref"
db_dirs["pitts30k"]="$datasets_vg_dir/pitts30k/test/database"
db_dirs["baidu_datasets"]="$datasets_vg_dir/baidu_datasets/training_images_undistort"
# Query directories for each dataset
declare -A qu_dirs
qu_dirs["Oxford"]="$datasets_vg_dir/Oxford_Robotcar/oxDataPart/2-s-resized"
qu_dirs["gardens"]="$datasets_vg_dir/gardens/night_right"
qu_dirs["st_lucia"]="$datasets_vg_dir/st_lucia/test/queries"
qu_dirs["17places"]="$datasets_vg_dir/17places/query"
qu_dirs["pitts30k"]="$datasets_vg_dir/pitts30k/test/queries"
qu_dirs["baidu_datasets"]="$datasets_vg_dir/baidu_datasets/query_images_undistort"
