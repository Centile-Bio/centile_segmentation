#!/bin/bash

# =============================================================================
# Script Name: submit_inference_scripts.sh
# Description: Submits inference jobs for specified dataset IDs using SLURM.
# Usage:        bash submit_inference_scripts.sh
# =============================================================================

# Array of Dataset IDs to process
DATASET_IDS=(22 24 27)

# Path to the inference SLURM script
INFERENCE_SCRIPT="inference_nnunet_bobs.sh"

# Check if the inference script exists and is executable
if [[ ! -f "$INFERENCE_SCRIPT" ]]; then
    echo "Error: Inference script '$INFERENCE_SCRIPT' not found in the current directory."
    exit 1
fi

if [[ ! -x "$INFERENCE_SCRIPT" ]]; then
    echo "Warning: Inference script '$INFERENCE_SCRIPT' is not executable. Attempting to set executable permission."
    chmod +x "$INFERENCE_SCRIPT"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to make '$INFERENCE_SCRIPT' executable."
        exit 1
    fi
    echo "Successfully set executable permission for '$INFERENCE_SCRIPT'."
fi

# Submit a SLURM job for each Dataset ID
for DATASET_ID in "${DATASET_IDS[@]}"
do
    echo "Submitting inference job for Dataset ID: $DATASET_ID"
    
    # Submit the job and capture the job ID
    JOB_ID=$(sbatch "$INFERENCE_SCRIPT" "$DATASET_ID" | awk '{print $4}')
    
    if [[ -n "$JOB_ID" ]]; then
        echo "Successfully submitted job $JOB_ID for Dataset ID $DATASET_ID."
    else
        echo "Error: Failed to submit job for Dataset ID $DATASET_ID."
    fi
done

echo "All inference jobs have been submitted."
