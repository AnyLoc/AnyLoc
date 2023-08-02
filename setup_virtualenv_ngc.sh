# Script to setup the virtual environment

echo "Python: $(which python)"
echo "Pip: $(which pip)"
read -p "Continue? [Ctrl-C to exit, enter to continue] "

function pip_install() {
    echo -ne "\e[0;36m"
    echo "pip install -U $@"
    echo -ne "\e[0m"
    pip install -U $@
}

# The following should be from the system NGC container
# - torch=='1.14.0a0+410ce96'
# - torchvision=='0.15.0a0'

# Install requirements
start_time=$(date)
echo "---- Start time: $start_time ----"
echo "---------- Installing core packages ----------"
# Could not find torchaudio version for NGC - can't use ImageBind :'(
#   Some notes (in case you want to figure it out...)
#       - https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html
#       - https://download.pytorch.org/whl/torchaudio/
#       - https://github.com/pytorch/pytorch/wiki/PyTorch-Versions
#       - https://pytorch.org/audio/main/installation.html
#       - Don't change torch, torchvision, or CUDA when installing this
# pip_install torchaudio=='0.15.0a0'
pip_install faiss-gpu=='1.7.2'
pip_install matplotlib=='3.6.2'
pip_install fast-pytorch-kmeans=='0.1.6'
pip_install einops=='0.6.0'
pip_install tqdm=='4.64.1'
pip_install joblib=='1.2.0'
pip_install wandb=='0.13.9'
pip_install natsort=='8.2.0'
pip_install scikit-learn=='0.24.2'
pip_install pandas=='2.0.0'
pip_install opencv-contrib-python-headless=='4.7.0.68'
pip_install tyro=='0.4.0'
pip_install scipy=='1.6.3'
pip_install imageio=='2.25.0'
pip_install seaborn=='0.12.1'
# TensorRT is overwriting pytorch install (don't want to do that!)
# pip_install torch-tensorrt=='1.3.0'
pip_install pytorchvideo=='0.1.5'
pip_install git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16
pip_install open-clip-torch=='2.16.0'
pip_install scikit-image=='0.19.3'
pip_install torchinfo=='1.7.2'
pip_install torchviz=='0.0.2'
pip_install jupyterlab=='2.3.2'
pip_install transformers=='4.28.0'
pip_install googledrivedownloader=='0.4'
pip_install nvitop=='1.0.0'
pip_install gpustat=='1.0.0'
pip_install distinctipy=='1.2.2'
pip_install torchscan=='0.1.2'
pip_install gradio=='3.37.0'

end_time=$(date)
echo "---- End time: $end_time ----"
echo "---- Start time was: $start_time ----"
# dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
# echo "--- Setup took (HH:MM:SS): `date -d@$dur -u +%H:%M:%S` ---"
echo "=========== Installation completed ==========="
