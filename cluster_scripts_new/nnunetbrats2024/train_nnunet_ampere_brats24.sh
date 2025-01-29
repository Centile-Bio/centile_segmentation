#!/bin/bash
#SBATCH -J brats24_nnunetv2
#SBATCH -A BETHLEHEM-SL2-GPU
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G               # Adjust as needed
#SBATCH --time=30:00:00
#SBATCH --output=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/logs/tr_brats_nnunetv2%j.out
#SBATCH --error=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/errors/tr_brats_nnunetv2%j.err

# Print the start of the job script
echo "=== Starting job script ==="

# Check if DATASET_ID is provided
if [ -z "$1" ]; then
  echo "Error: No dataset ID provided. Please provide the dataset ID as an argument."
  exit 1
fi

# Set the dataset ID variable
DATASET_ID=$1

# Print the dataset ID
echo "Dataset ID: $DATASET_ID"

# Generate dataset path strings
DATASET_INPUT_PATH="/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_raw/Dataset0${DATASET_ID}_brats24/imagesTs"
DATASET_OUTPUT_PATH="/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_results/Dataset0${DATASET_ID}_brats24"

# Confirm paths
echo "Input Path: $DATASET_INPUT_PATH"
echo "Output Path: $DATASET_OUTPUT_PATH"

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

# Set nnU-Net environment variables
echo "Setting nnU-Net environment variables..."
export nnUNet_raw=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_raw
export nnUNet_preprocessed=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_preprocessed
export nnUNet_results=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_results

# Confirm environment variables are set
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"

# Print current working directory
echo "Current working directory:"
pwd

# Print Python version to confirm environment
echo "Python version:"
python --version

# Change to the project root directory
echo "Changing to project root directory..."
cd /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission

# Confirm current directory
echo "Now in directory:"
pwd

# Run the training command
echo "Starting training command..."

# Plan and preprocess the data
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c 3d_fullres

# Train
nnUNetv2_train $DATASET_ID 3d_fullres all

# Inference (Uncomment if needed)
# nnUNetv2_predict -i $DATASET_INPUT_PATH -o $DATASET_OUTPUT_PATH -d $DATASET_ID -c 3d_fullres -f all

# Print the completion of the job script
echo "=== Job completed ==="
