# Run the CLIP experiments (without and with PCA)

# --------------------- Paths ---------------------
# Cache directory (where images and model cache will be stored)
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Dataset path (where the datasets are downloaded)
ds_path="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# ----------------- General parameters -----------------
# The datasets downloaded (for testing)
datasets=("st_lucia" "pitts30k" "17places")
# All the CLIP implementation to test
clip_impls=(    # CLIP Implementations: Impl, Backbone, Pretrained  # VRAM
    "openai, ViT-B/32, None"      # < 6 GB
    "openai, RN50x64, None"     # < 6 GB
    "openai, ViT-L/14@336px, None"      # < 6 GB
    "open_clip, RN101-quickgelu, openai"        # < 6 GB
    "open_clip, coca_ViT-L-14, mscoco_finetuned_laion2b_s13b_b90k"  # < 6 GB
    "open_clip, ViT-g-14, laion2b_s12b_b42k"        # < 12 GB
    "open_clip, ViT-bigG-14, laion2b_s39b_b160k"    # < 16 GB
    "open_clip, convnext_large_d_320, laion2b_s29b_b131k_ft_soup"   # < 12 GB
    "open_clip, ViT-H-14, laion2b_s32b_b79k"        # < 12 GB
    "open_clip, xlm-roberta-large-ViT-H-14, frozen_laion5b_s13b_b90k" # < 12 GB
)
# Lower dimensionality for PCA (None means no PCA)
# pca_dims=("32" "64" "128" "256")  # If using multi
pca_dim="None"
# Percentage of components from lower eigen-basis
# pca_lowfs=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0")   # If using multi
pca_lowf="0.0"
# Other minor parameters
exp_prefix="many_run"       # A prefix for the experiment ID (when caching)
top_k_vals="{1..20}"        # Top-k values for recalls
percent_qual_save="0.025"   # Percentage of queries (for qualitative results)
# ----------- Weights and Biases parameters -----------
wandb_project="CLIP-PCA"        # Project name (set to "None" for no WandB)
# wandb_project="None"            # Project name (set to "None" for no WandB)
wandb_group="direct_clip"       # Group (for runs)

# Wait for older experiments to get over
# del_time=10800
# echo "Sleeping for $del_time seconds before starting!"
# sleep $del_time;

n=${#clip_impls[*]}
for ((i=0; i<$n; i++)) do
    IFS=", " read clip_impl clip_backbone clip_pretrained <<< ${clip_impls[$i]}
    printf "============> %-10s --- %-20.20s --- %-20.20s <============\n" \
        $clip_impl $clip_backbone $clip_pretrained
    # for pca_dim in ${pca_dims[*]}; do
    # for pca_lowf in ${pca_lowfs[*]}; do
    for ds in ${datasets[*]}; do
        exp_id=$(head /dev/urandom | md5sum -b | awk '{print substr($1,0,4)}')
        exp_id="$exp_prefix/$exp_id"
        while [[ -d $exp_id ]]; do
            echo "Experiment folder already exists, trying a new one"
            exp_id=$(head /dev/urandom | md5sum -b | \
                awk '{print substr($1,0,4)}')
            exp_id="$exp_prefix/$exp_id"
        done
        # Construct the python command
        python_cmd="python ./scripts/clip_top_k_vpr.py"
        python_cmd="$python_cmd --exp-id $exp_id --clip-impl $clip_impl"
        python_cmd="$python_cmd --clip-backbone $clip_backbone"
        if [[ $clip_impl == "open_clip" ]]; then
            python_cmd="$python_cmd --clip-pretrained $clip_pretrained"
        fi
        python_cmd="$python_cmd --top-k-vals $top_k_vals"
        python_cmd="$python_cmd --qual-result-percent $percent_qual_save"
        python_cmd="$python_cmd --prog.cache-dir $cache_dir"
        python_cmd="$python_cmd --prog.data-vg-dir $ds_path"
        python_cmd="$python_cmd --prog.vg-dataset-name $ds"
        if [[ $pca_dim != "None" ]]; then
            python_cmd="$python_cmd --pca-dim-reduce $pca_dim"
            python_cmd="$python_cmd --pca-low-factor $pca_lowf"
        fi
        if [[ $wandb_project == "None" ]]; then
            python_cmd="$python_cmd --prog.no-use-wandb"
        else
            python_cmd="$python_cmd --prog.wandb-proj $wandb_project"
            python_cmd="$python_cmd --prog.wandb-group $wandb_group"
        fi
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo -ne "\e[0;36m"
        echo $python_cmd
        echo -ne "\e[0m"
        eval $python_cmd    # Main execution
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    done
    # done
    # done
done
