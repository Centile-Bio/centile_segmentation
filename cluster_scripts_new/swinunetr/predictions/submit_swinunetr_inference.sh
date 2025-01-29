#!/bin/bash

# ======================================================================
# Script Name: submit_swinunetr_inference
# Description: Submits predefined SwinUNETR inference scripts to the SLURM
#              job scheduler.
# Usage: ./submit_swinunetr_inference
# ======================================================================

# Set the directory containing the inference scripts
SCRIPT_DIR="/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_scripts_new/swinunetr/predictions"

# Change to the script directory
cd "$SCRIPT_DIR" || { echo "Error: Cannot change directory to $SCRIPT_DIR"; exit 1; }

# Optional: Load necessary modules or activate environments
# Uncomment and modify the lines below if needed
# module load python/3.8
# source /path/to/your/virtualenv/bin/activate

# Array of script names to submit
SCRIPTS=(
    "inference_swinunetr_from_backbone_checkpoint.sh"
    "inference_swinunetr_from_backbone_checkpoint_ukb_fine_tuned_from_ct.sh"
    "inference_swinunetr_from_backbone_checkpoint_ukb_from_scratch.sh"
    "inference_swinunetr_from_brats24_checkpoint.sh"
    "inference_swinunetr_from_scratch.sh"
)

# Submit each script individually
for script in "${SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        if [[ -x "$script" ]]; then
            echo "Submitting $script to SLURM..."
            sbatch "$script"

            # Check if submission was successful
            if [[ $? -ne 0 ]]; then
                echo "Warning: Failed to submit $script"
            else
                echo "Successfully submitted $script"
            fi
        else
            echo "Making $script executable..."
            chmod +x "$script"
            echo "Submitting $script to SLURM..."
            sbatch "$script"

            # Check if submission was successful
            if [[ $? -ne 0 ]]; then
                echo "Warning: Failed to submit $script"
            else
                echo "Successfully submitted $script"
            fi
        fi
    else
        echo "Error: Script $script does not exist in $SCRIPT_DIR"
    fi
done

echo "All specified scripts have been processed for submission."
