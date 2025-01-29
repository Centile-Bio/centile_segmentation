#!/bin/bash

echo "=== Submitting jobs for dataset IDs 10 through 19 ==="

# Loop from 10 to 19 inclusive
for DATASET_ID in {15..19}; do
    echo "Submitting job for Dataset ID: $DATASET_ID"
    sbatch /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/cluster_scripts_new/nnunetbrats2024/train_nnunet_ampere_brats24.sh "$DATASET_ID"
    
    # Check if sbatch submission was successful
    if [ $? -eq 0 ]; then
        echo "Successfully submitted job for Dataset ID: $DATASET_ID"
    else
        echo "Failed to submit job for Dataset ID: $DATASET_ID"
    fi
    
    echo "-------------------------------------"
done

echo "=== All jobs submitted ==="
