#!/bin/bash
#SBATCH -J train_synthseg
#SBATCH -A BETHLEHEM-SL2-GPU
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/logs/train_synthseg%j.out
#SBATCH --error=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/errors/train_synthseg%j.err

echo "=== Starting job script ==="

# Load necessary modules
echo "Loading modules..."
module purge
module load cuda/10.1

# checking that cuda is loaded
cuda-smi

# Activate environment
source /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/synthseg/bin/activate

# Check the Python version
python --version

# Enter the correct folder
cd /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/SynthSeg/scripts/tutorials

# Set the correct PYTHONPATH
export PYTHONPATH=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/SynthSeg/:$PYTHONPATH

# Confirm current directory
echo "Now in directory:"
pwd

# Check GPU memory status before starting
echo "Checking initial GPU memory status:"
nvidia-smi

# Run the training command
echo "Starting training command..."
python 3-training.py

echo "=== Job completed ==="
