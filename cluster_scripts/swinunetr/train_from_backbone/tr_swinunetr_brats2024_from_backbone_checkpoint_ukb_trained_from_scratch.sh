#!/bin/bash
#SBATCH -J tr_swinunetr_brats2024_from_backbone_checkpoint_ukb_trained_from_scratch
#SBATCH -A BETHLEHEM-SL2-GPU
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G               # Adjust as needed
#SBATCH --time=30:00:00
#SBATCH --output=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/logs/tr_swinunetr_brats2024_from_backbone_checkpoint_ukb_trained_from_scratch%j.out
#SBATCH --error=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/errors/tr_swinunetr_brats2024_from_backbone_checkpoint_ukb_trained_from_scratch%j.err

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

/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/conda_envs/swinunetr/bin/python /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/BRATS21/main.py \
    --json_list /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/brats2024/brats-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/brats2024.json \
    --data_dir /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/brats2024/brats-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth \
    --val_every 5 \
    --noamp \
    --roi_x 128 \
    --roi_y 128 \
    --roi_z 128 \
    --in_channels 4 \
    --out_channels 3 \
    --spatial_dims 3 \
    --use_checkpoint \
    --feature_size 48 \
    --resume_ckpt \
    --logdir tr_from_backbone_checkpoint_ukb_from_scratch \
    --pretrained_dir /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/Pretrain/runs/pretrain_swinunetr_ukb_from_scratch \
    --pretrained_model_name model_bestValRMSE.pt \
    --use_ssl_pretrained_backbone \
    --save_checkpoint

# Print the completion of the job script
echo "=== Job completed ==="
