#!/bin/bash
#SBATCH -J inf_swinunet
#SBATCH -A BETHLEHEM-SL2-GPU
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G               # Adjust as needed
#SBATCH --time=30:00:00
#SBATCH --output=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/logs/inference_swinunetr_from_brats24_checkpoint%j.out
#SBATCH --error=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/errors/inference_swinunetr_from_brats24_checkpoint%j.err

# Print the start of the job script
echo "=== Starting job script ==="

# Load necessary modules
echo "Loading modules..."
module purge
module load cuda/11.8            # Load CUDA 11.8 as per PyTorch version
module load miniconda/3          # Load Conda module

# Initialize Conda
echo "Initializing Conda..."
eval "$(conda shell.bash hook)"

# Manually specify the Python path to ensure correct environment is activated
export PYTHONPATH="/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/conda_envs/swinunetr/bin:$PYTHONPATH"

# Activate your Conda environment
echo "Activating Conda environment: swinunetr"
conda activate /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/conda_envs/swinunetr

# Verify Conda environment
echo "Current Conda environment:"
conda info --envs

# print which python is being used
echo "Using Python binary at:"
which python

# Print Python version to confirm environment
echo "Python version:"
python --version

# Change to the project root directory
echo "Changing to project root directory..."
cd /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/BRATS21

# Confirm current directory
echo "Now in directory:"
pwd

# Run the training command
echo "Starting training command..."

/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/conda_envs/swinunetr/bin/python /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/BRATS21/predict.py \
    --json_list /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/brats2024/brats-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/brats2024.json \
    --data_dir /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/brats2024/brats-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth \
    --batch_size=1 \
    --noamp \
    --use_checkpoint \
    --pretrained_dir /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/BRATS21/runs/tr_from_brats2024_checkpoint \
    --pretrained_model_name model.pt \


/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/conda_envs/swinunetr/bin/python /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/BRATS21/predict.py \
    --json_list /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/brats2024/brats-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/brats2024.json \
    --data_dir /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/brats2024/brats-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth \
    --batch_size=1 \
    --noamp \
    --use_checkpoint \
    --pretrained_dir /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/BRATS21/runs/tr_from_brats2024_checkpoint \
    --pretrained_model_name model_final.pt \

# Print the completion of the job script
echo "=== Job completed ==="
