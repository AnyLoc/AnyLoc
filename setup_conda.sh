# Script to setup conda environment

_CONDA_ENV_NAME="${1:-anyloc}"

# Ensure conda is installed
if ! [ -x "$(command -v conda)" ]; then
    echo 'Error: conda is not installed. Source or install Anaconda'
    exit 1
fi

# Ensure environmnet
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo 'No conda environment activated'
    exit 1
fi
if [ "$CONDA_DEFAULT_ENV" != "$_CONDA_ENV_NAME" ]; then
    echo "Wrong conda environment activated. Activate $_CONDA_ENV_NAME"
    exit 1
fi

# Confirm environment
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
read -p "Continue? [Ctrl-C to exit, enter to continue] "

# Functions
function conda_install() {
    echo -ne "\e[0;36m"
    echo "conda install -y --freeze-installed --no-update-deps $@"
    echo -ne "\e[0m"
    conda install -y --freeze-installed --no-update-deps $@
}
function pip_install() {
    echo -ne "\e[0;36m"
    echo "pip install --upgrade $@"
    echo -ne "\e[0m"
    pip install --upgrade $@
}

# Install requirements
start_time=$(date)
echo "---- Start time: $start_time ----"
echo "---------- Installing core packages ----------"
conda_install  -c pytorch -c nvidia pytorch==1.13.1 \
    torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7
conda_install -c pytorch faiss-gpu
conda_install -c conda-forge matplotlib
pip_install pyqt5   # For matplotlib
pip_install fast-pytorch-kmeans
conda_install -c conda-forge einops
conda_install -c conda-forge tqdm
conda_install -c conda-forge joblib
conda_install -c conda-forge wandb
conda_install -c conda-forge natsort
conda_install -c conda-forge scikit-learn
conda_install -c conda-forge pandas
pip_install opencv-python==4.3.0.36 # This version doesn't break matplotlib
pip_install tyro
conda_install -c conda-forge scipy
conda_install -c conda-forge imageio
conda_install -c conda-forge seaborn
pip_install torch-tensorrt
pip_install pytorchvideo
conda_install -c conda-forge transformers
conda_install -c conda-forge googledrivedownloader
conda_install -c conda-forge distinctipy
echo "---- Installing CLIP ----"
pip_install git+https://github.com/openai/CLIP.git
pip_install open_clip_torch
echo "---- Installing more packages ----"
conda_install -c conda-forge scikit-image
conda_install -c conda-forge torchinfo
conda_install -c conda-forge graphviz
conda_install -c conda-forge gradio
pip_install torchviz
pip_install torchscan
pip_install onedrivedownloader
echo "---------- Installing other packages ----------"
conda_install -c conda-forge jupyter
conda_install -c conda-forge nvitop
conda_install -c conda-forge gpustat
pip_install webm
pip_install "imageio[ffmpeg]"

end_time=$(date)
echo "---- End time: $end_time ----"
dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
echo "--- Setup took (HH:MM:SS): `date -d@$dur -u +%H:%M:%S` ---"
echo "----- Environment $CONDA_DEFAULT_ENV has been setup -----"
