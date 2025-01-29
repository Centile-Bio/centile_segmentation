#!/bin/bash
#SBATCH -J inference_biomed_parse
#SBATCH -A BETHLEHEM-SL2-GPU
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16G               # Adjust as needed
#SBATCH --time=00:01:00
#SBATCH --output=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/logs/inference_biomed_parse%j.out
#SBATCH --error=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/errors/inference_biomed_parse%j.err

# Load necessary modules
echo "Loading modules..."
module purge
module load cuda/11.8            # Load CUDA 11.8 as per PyTorch version
module load miniconda/3          # Load Conda module

# Specify the correct Python path manually
PYTHON_PATH="$HOME/.conda/envs/mamba_env/envs/biomedparse/bin/python"

# Verify the Python version being used
echo "Using Python from: $PYTHON_PATH"
$PYTHON_PATH --version

# Change to the project root directory
echo "Changing to project root directory..."
cd /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/BiomedParse

# Confirm current directory
echo "Now in directory:"
pwd

# Run the prediction script
echo "Starting prediction script..."
$PYTHON_PATH my_prediction.py

echo "=== Job completed ==="
