#!/bin/bash
#SBATCH -J pretrain_swinunetr_ukb_from_scratch
#SBATCH -A BETHLEHEM-SL2-GPU
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64G            
#SBATCH --time=30:00:00
#SBATCH --output=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/logs/pretrain_swinunetr_ukb_from_scratch%j.out
#SBATCH --error=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/errors/pretrain_swinunetr_ukb_from_scratch%j.err

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
export PYTHONPATH="/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/conda_envs/swinunetr_env_py39/bin:$PYTHONPATH"

# Activate your Conda environment
echo "Activating Conda environment: swinunetr_env_py39"
conda activate /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/conda_envs/swinunetr_env_py39

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
cd /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/Pretrain

# Confirm current directory
echo "Now in directory:"
pwd

# Run the training command
echo "Starting training command..."

/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/conda_envs/swinunetr_env_py39/bin/torchrun \
    --nproc_per_node=1 \
    --master_port=11223 \
    /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/research-contributions/SwinUNETR/Pretrain/main.py \
    --json_list /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/synthseg_training_labels/ukb_train_for_swin_unetr_backbone_pretraining.json \
    --data_dir /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/synthseg_training_labels \
    --logdir pretrain_swinunetr_ukb_from_scratch \
    --batch_size=2 \
    --num_steps=100000 \
    --lrdecay \
    --eval_num=500 \
    --lr=6e-6 \
    --decay=0.1

# Print the completion of the job script
echo "=== Job completed ==="
