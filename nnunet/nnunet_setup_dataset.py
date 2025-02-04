import os
import argparse
import shutil

def create_dataset_folder_structure(base_path, dataset_name, dataset_id, overwrite=False):
    """
    Create the folder structure required for an nnU-Net dataset in multiple locations.

    Args:
        base_path (str): The base directory containing nnUNet_raw, nnUNet_preprocessed, nnUNet_results.
        dataset_name (str): The dataset name (e.g., BrainTumour).
        dataset_id (int): The dataset ID (an integer between 1 and 999).
        overwrite (bool): If True, existing folders will be deleted and recreated.

    Example:
        python nnunet_setup_dataset.py --base_path /path/to/nnUNet --dataset_name synthseg --dataset_id 7 --overwrite
    """
    # Validate dataset_id
    if not isinstance(dataset_id, int) or dataset_id < 1 or dataset_id > 999:
        raise ValueError("dataset_id must be an integer between 1 and 999.")
    
    dataset_id_str = f"Dataset{dataset_id:03d}_{dataset_name}"

    # Define the locations
    locations = {
        "nnUNet_raw": os.path.join(base_path, "nnUNet_raw", dataset_id_str),
        "nnUNet_preprocessed": os.path.join(base_path, "nnUNet_preprocessed", dataset_id_str),
        "nnUNet_results": os.path.join(base_path, "nnUNet_results", dataset_id_str),
    }

    for location_name, dataset_path in locations.items():
        # Handle overwrite option
        if overwrite and os.path.exists(dataset_path):
            print(f"Overwriting existing dataset folder at: {dataset_path}")
            shutil.rmtree(dataset_path)
        
        # Create folders (only nnUNet_raw has subdirectories)
        if location_name == "nnUNet_raw":
            os.makedirs(os.path.join(dataset_path, 'imagesTr'), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, 'imagesTs'), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, 'labelsTr'), exist_ok=True)
        else:
            os.makedirs(dataset_path, exist_ok=True)

        print(f"Folder structure created at: {dataset_path}")

def main():
    parser = argparse.ArgumentParser(description="Create nnU-Net dataset folder structure in all required locations.")
    parser.add_argument('--base_path', required=True, type=str, help='The base directory containing nnUNet folders (e.g., /path/to/nnUNet).')
    parser.add_argument('--dataset_name', required=True, type=str, help='The dataset name (e.g., BrainTumour).')
    parser.add_argument('--dataset_id', required=True, type=int, help='The dataset ID (an integer between 1 and 999).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset folders.')
    args = parser.parse_args()
    
    create_dataset_folder_structure(args.base_path, args.dataset_name, args.dataset_id, args.overwrite)

if __name__ == "__main__":
    main()
