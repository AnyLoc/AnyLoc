# Setup everything for this repository

readonly ARGS="$@"  # Reset using https://stackoverflow.com/a/4827707
readonly PROGNAME=$(basename $0)
readonly PROGPATH=$(realpath $(dirname $0))

# Internal variables
env_name="anyloc"   # Name of the environment
exec_name="conda"           # Executable
dry_run="false"     # 'true' or 'false'
ask_prompts="true"  # 'true' or 'false'
dev_tools="false"   # 'true' or 'false'
warn_exit="true"    # 'true' or 'false'

# Output formatting
debug_msg_fmt="\e[2;90m"
info_msg_fmt="\e[1;37m"
warn_msg_fmt="\e[1;35m"
fatal_msg_fmt="\e[2;31m"
command_msg_fmt="\e[0;36m"
# Wrapper printing functions
echo_debug () {
    echo -ne $debug_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
echo_info () {
    echo -ne $info_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
echo_warn () {
    echo -ne $warn_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
echo_fatal () {
    echo -ne $fatal_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
echo_command () {
    echo -ne $command_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
# Installer functions
function run_command() {
    echo_command $@
    if [ $dry_run == "true" ]; then
        echo_debug "Dry run, not running command..."
    else
        $@
    fi
}
function conda_install() {
    run_command $exec_name install -y --freeze-installed --no-update-deps $@
    ec=$?
    if [[ $ec -gt 0 ]]; then
        echo_warn "Could not install '$@', maybe try though conda_raw_install"
        if [[ $warn_exit == "true" ]]; then
            exit $ec
        else
            echo_debug "Exit on warning not set, continuing..."
        fi
    fi
}
function conda_raw_install() {
    run_command $exec_name install -y $@
}
function pip_install() {
    run_command pip install --upgrade $@
}

# Ensure installation can happen
if [ -x "$(command -v mamba)" ]; then   # If mamba found
    echo_debug "Found mamba"
    exec_name="mamba"
elif [ -x "$(command -v conda)" ]; then # If conda found
    echo_debug "Found conda (couldn't find mamba)"
    exec_name="conda"
else
    echo_fatal "Could not find mamba or conda! Install, source, and \
            activate it."
    exit 127
fi

function usage() {
    cat <<-EOF

Environment setup for AnyLoc

Usage: 
    1. bash $PROGNAME [-OPTARG VAL ...]
    2. bash $PROGNAME --help
    3. bash $PROGNAME NAME [-OPTARG VAL ...]

All optional arguments:
    -c | --conda INST       Conda installation ('mamba' or 'conda'). By
                            default, 'mamba' is used (if installed), else
                            'conda'.
    -d | --dev              If passed, the documentation and packaging 
                            tools are also installed (they aren't, by 
                            default). These are only for developers.
        --dry-run           If passed, the commands are printed instead of
                            running them.
    -e | --env-name NAME    Name of the conda/mamba environment. This can
                            also be passed as the 1st positional argument.
    -h | --help             Show this message.
        --no-exit-on-warn   By default, a warning causes the script to
                            exit (with a suggestion modification). If this
                            option is passed, the script doesn't exit (it
                            continues).
    -n | --no-prompt        By default, a prompt is shown (asking to press
                            Enter to continue). If this is passed, the
                            prompt is not shown.

Exit codes
    0       Script executed successfully
    1       Argument error (some wrong argument was passed)
    127     Could not find conda or mamba (executable)
    -       Some warning (if exit on warning)
EOF
}

function parse_options() {
    # Set passed arguments
    set -- $ARGS
    pos=1
    while (( "$#" )); do
        arg=$1
        shift
        case "$arg" in
            # Conda installation to use
            "--conda" | "-c")
                ci=$1
                shift
                echo_debug "Using $ci (for anaconda base)"
                exec_name=$ci
                ;;
            # Developer install options
            "--dev" | "-d")
                echo_debug "Installing documentation and packaging tools"
                dev_tools="true"
                ;;
            # Dry run
            "--dry-run")
                echo_debug "Dry run mode enabled"
                dry_run="true"
                ;;
            # Environment name
            "--env-name" | "-e")
                en=$1
                shift
                echo_debug "Using environment $en"
                env_name=$en
                ;;
            # Help options
            "--help" | "-h")
                usage
                exit 0
                ;;
            # No exit on warning
            "--no-exit-on-warn")
                echo_debug "No exit on warning set"
                warn_exit="false"
                ;;
            # No prompt
            "--no-prompt" | "-n")
                echo_debug "Not showing prompts (no Enter needed)"
                ask_prompts="false"
                ;;
            *)
                if [ $pos -eq 1 ]; then # Environment name
                    echo_debug "Using environment $arg"
                    env_name=$arg
                else
                    echo_fatal "Unrecognized option: $arg"
                    echo_debug "Run 'bash $PROGNAME --help' for usage"
                    exit 1
                fi 
        esac
        pos=$((pos + 1))
    done
}

# ====== Main program entrypoint ======
parse_options
if [ -x "$(command -v $exec_name)" ]; then
    echo_info "Using $exec_name (for base anaconda)"
else
    echo_fatal "Could not find $exec_name! Install, source, and \
            activate it."
    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" != "$env_name" ]; then
    echo_fatal "Wrong environment activated. Activate $env_name"
    exit 1
fi

# Confirm environment
echo_info "Using environment: $CONDA_DEFAULT_ENV"
echo_info "Python: $(which python)"
echo_debug "Python version: $(python --version)"
echo_info "Pip: $(which pip)"
echo_debug "Pip version: $(pip --version)"
if [ $ask_prompts == "true" ]; then
    read -p "Continue? [Ctrl-C to exit, enter to continue] "
elif [ $ask_prompts == "false" ]; then
    echo_info "Continuing..."
fi

# Install packages
start_time=$(date)
start_time_secs=$SECONDS
echo_debug "---- Start time: $start_time ----"
# Core packages using conda_install and conda_raw_install
echo_info "------ Installing core packages ------"
conda_raw_install  -c pytorch -c nvidia pytorch==1.13.1 \
    torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7
conda_raw_install -c pytorch faiss-gpu==1.7.2
conda_raw_install -c conda-forge matplotlib==3.6.2
pip_install fast-pytorch-kmeans==0.1.6
conda_raw_install -c conda-forge einops==0.6.0
conda_raw_install -c conda-forge tqdm==4.64.1
conda_raw_install -c conda-forge joblib==1.2.0
conda_raw_install -c conda-forge wandb==0.13.9
conda_raw_install -c conda-forge natsort==8.2.0
conda_raw_install -c conda-forge pandas==2.0.0
conda_raw_install -c conda-forge opencv=4.7
conda_raw_install -c conda-forge tyro
conda_raw_install -c conda-forge scipy==1.6.3
conda_raw_install -c conda-forge scikit-learn==0.24.2
conda_raw_install -c conda-forge imageio==2.25.0
conda_raw_install -c conda-forge seaborn==0.12.1
# pip_install torch-tensorrt  # This replaces torch with '2.0.1+cu117'
pip_install pytorchvideo==0.1.5
conda_raw_install -c conda-forge transformers==4.28.0
conda_raw_install -c conda-forge googledrivedownloader==0.4
conda_raw_install -c conda-forge distinctipy==1.2.2
echo_info "------ Installing CLIP ------"
pip_install git+https://github.com/openai/CLIP.git
pip_install open-clip-torch==2.16.0
echo_info "------ Installing additional packages ------"
conda_raw_install -c conda-forge scikit-image==0.19.3
conda_raw_install -c conda-forge torchinfo==1.7.2
conda_raw_install -c conda-forge graphviz
conda_raw_install -c conda-forge gradio
pip_install torchviz=='0.0.2'
pip_install torchscan
pip_install onedrivedownloader
# Core packages using pip_install
if [ $dev_tools == "true" ]; then 
    echo_info "------ Installing documentation and packaging tools ------"
    conda_raw_install -c conda-forge jupyter
    conda_raw_install -c conda-forge nvitop
    conda_raw_install -c conda-forge gpustat
    pip_install webm
    pip_install "imageio[ffmpeg]"
    # Other packages (only for development)
elif [ $dev_tools == "false" ]; then
    echo_info "Skipping documentation and packaging tools"
fi

# Installation ended
end_time=$(date)
end_time_secs=$SECONDS
echo_debug "---- End time: $end_time ----"
# dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
dur=$(( $end_time_secs - $start_time_secs ))
_d=$(( dur/3600/24 ))   # Days!
echo_info "---- Environment setup took (d-HH:MM:SS): \
        $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo_info "----- Environment $CONDA_DEFAULT_ENV has been setup -----"
echo_debug "Starting time: $start_time"
echo_debug "Ending time: $end_time"
