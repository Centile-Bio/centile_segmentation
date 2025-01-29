#!/bin/bash
#SBATCH -J bobs24_nnunetv2_inference
#SBATCH -A BETHLEHEM-SL2-GPU
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16G               # Reduced memory if inference requires less
#SBATCH --time=10:00:00         # Adjusted time for inference tasks
#SBATCH --output=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/logs/inference_bobs_nnunetv2%j.out
#SBATCH --error=/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_results/errors/inference_bobs_nnunetv2%j.err

# Print the start of the job script
echo "=== Starting Inference Job Script ==="

# **Define dataset ID from command-line argument**
if [ -z "$1" ]; then
    echo "Error: No DATASET_ID provided."
    echo "Usage: sbatch run_nnunetv2_inference.sh <DATASET_ID>"
    exit 1
fi

DATASET_ID=$1

# Print the dataset ID
echo "Dataset ID: $DATASET_ID"

# Generate dataset path strings
DATASET_INPUT_PATH="/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_raw/Dataset0${DATASET_ID}_bobs/imagesTs"    
DATASET_OUTPUT_PATH="/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_results/Dataset0${DATASET_ID}_bobs/inferences"

# Confirm paths
echo "Input Path: $DATASET_INPUT_PATH"
echo "Output Path: $DATASET_OUTPUT_PATH"

# Load necessary modules
echo "Loading modules..."
module purge
module load cuda/11.8            # Ensure compatibility with your environment
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

# Run the inference command
echo "Starting inference command..."

nnUNetv2_predict \
    -i $DATASET_INPUT_PATH \
    -o $DATASET_OUTPUT_PATH \
    -d $DATASET_ID \
    -c 3d_fullres \
    -f all

# Print the completion of the job script
echo "=== Inference Job Completed ==="
