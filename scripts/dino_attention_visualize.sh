# Visualize the Dino attention maps through the CLS token.
#
# Usage:
#  bash scripts/dino_attention_visualize.sh DS_STRETCH FREQ
#
# DS_STRETCH 
#   Dataset stretch to use. This is the name of the dataset directory.
#   examples: '17places/ref', 'pitts30k/images/test/database'
# FREQ
#   Frequency of image sampling (every N-th image). Default: 10
# 


ds_dir="/scratch/avneesh.mishra/vl-vpr/datasets"
ds_stretch="${1:-17places/ref}"
s_freq="${2:-10}"
export CUDA_VISIBLE_DEVICES=${3:-0}

# Dataset directory must exist
if [ ! -d $ds_dir ]; then
    echo "Dataset directory does not exist: $ds_dir"
    exit 1
fi

# Check if symlink exists (make if not)
if [ ! -e "./dino_repo_main" ]; then
    echo "Creating symlink to original Dino repo"
    repo_cd="$HOME/.cache/torch/hub/facebookresearch_dino_main"
    if [ ! -e "$repo_cd" ]; then
        echo "Repo could not be found at '$repo_cd'."
        exit 1
    else
        ln -s "$repo_cd" "./dino_repo_main"
        echo "Made symlink"
    fi
else
    echo "Found the dino repository symlinked"
fi

# Output directory
out_dir="/scratch/$USER/out/dino/attentions/$ds_stretch"
if [ ! -d "$out_dir" ]; then
    mkdir -p "$out_dir"
    echo "Created output directory: $out_dir"
else
    echo "Output directory already exists: $out_dir"
fi

# Get all files
num_files=$(ls -1 "${ds_dir}/${ds_stretch}" | wc -l)
to_run=$(( num_files/s_freq ))
echo "To run: $to_run"
# SAVEIFS=$IFS
# IFS=$'\n'
all_files=($(ls -1 "${ds_dir}/${ds_stretch}" | natsort | xargs))
# IFS=$SAVEIFS
echo "Number of files in dataset: $num_files"
proc_files=0

# Run the visualization script for all images
cd dino_repo_main
echo "In directory: $(pwd)"
for ((i=0; i<$num_files; i+=$s_freq)); do
    # echo "Processing file: $i"
    proc_files=$(( proc_files+1 ))
    fname="${all_files[$i]}"
    bn="$(basename $fname)"
    echo "Processing file: ${bn%.*}"
    python ./visualize_attention.py --arch vit_small --patch_size 16 \
        --image_path "${ds_dir}/${ds_stretch}/${fname}" \
        --image_size 224 298 \
        --output_dir "${out_dir}/image-${bn%.*}"
done
cd ../
echo "Back in directory: $(pwd)"
echo "Processed $proc_files files"

echo "Program finished"

