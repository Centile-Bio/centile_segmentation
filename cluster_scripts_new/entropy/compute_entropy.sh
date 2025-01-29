#!/bin/bash
#SBATCH -J entropy
#SBATCH -A BETHLEHEM-SL2-GPU
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16G               # Adjust as needed
#SBATCH --time=02:00:00
#SBATCH --output=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/logs/entropy_%j.out
#SBATCH --error=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/errors/entropy_%j.err

echo "=== Starting job script ==="


# Load necessary modules
echo "Loading modules..."
module purge
module load cuda/11.8            # Load CUDA 11.8 as per PyTorch version
module load miniconda/3          # Load Conda module

# Initialize Conda
echo "Initializing Conda..."
eval "$(conda shell.bash hook)"

# Activate your Conda environment
echo "Activating Conda environment: nnunetv2"
conda activate nnunetv2

# Verify Conda environment
echo "Current Conda environment:"
conda info --envs
# print which python is being used
which python


# Print Python version to confirm environment
echo "Python version:"
python --version

# Change to the project root directory
echo "Changing to project root directory..."
cd /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/entropy_mindboggle_dataset008

/home/fp427/.conda/envs/nnunetv2/bin/python compute_mindboggle_entropy.py

echo "=== Job completed ==="
