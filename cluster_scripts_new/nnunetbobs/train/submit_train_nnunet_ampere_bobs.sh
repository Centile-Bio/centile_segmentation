#!/bin/bash

# Array of DATASET_IDs to submit, from 20 to 28

DATASET_IDS=(20 21 22 23 24 25 26 27 28)

# Path to the parameterized job script
JOB_SCRIPT="train_nnunet_ampere_bobs.sh"

# Check if the job script exists and is executable
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: Job script '$JOB_SCRIPT' not found."
    exit 1
fi

if [ ! -x "$JOB_SCRIPT" ]; then
    echo "Job script '$JOB_SCRIPT' is not executable. Setting execute permission."
    chmod +x "$JOB_SCRIPT"
fi

# Loop through each DATASET_ID and submit the job
for ID in "${DATASET_IDS[@]}"; do
    echo "Submitting job for DATASET_ID=$ID..."
    sbatch "$JOB_SCRIPT" "$ID"
done

echo "All jobs submitted."
