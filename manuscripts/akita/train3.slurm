#!/bin/bash
#SBATCH --job-name=akita_train
#SBATCH --output=akita_train2.out
#SBATCH --error=akita_train2.err
#SBATCH --partition=mgpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
# Load modules (ensure correct CUDA version)
module unload cuda/12.6
module unload nvhpc-hpcx-cuda11/24.11
module load nvhpc-hpcx-cuda11/24.11
# Load Conda properly (check for correct path)
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust if needed
conda activate basenji  # Activate Conda environment

export CUDA_HOME=$HOME/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH



# Create output directory if missing
mkdir -p ./data/1m/train_out/

# Debugging: Print GPU availability
which nvcc
nvcc --version
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# Run training script
python /home/017448899/basenji_thesis/bin/akita_train.py -k -o ./data/1m/train_out/ ./data/1m/params2.json ./data/1m