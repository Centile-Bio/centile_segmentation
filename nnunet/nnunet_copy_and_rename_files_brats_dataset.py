#!/usr/bin/env python3

import os
import argparse
import shutil
import nibabel as nib
from tqdm import tqdm
import json

def get_nnunet_dataset_paths(nnunet_raw_path, dataset_id, dataset_name):
    """
    Given nnunet_raw_path, dataset_id, and dataset_name, returns the paths to imagesTr, imagesTs, and labelsTr directories.

    Args:
        nnunet_raw_path (str): The path to nnUNet_raw directory.
        dataset_id (int): The dataset ID (an integer between 1 and 999).
        dataset_name (str): The dataset name (e.g., 'brats').

    Returns:
        tuple: imagesTr_path, imagesTs_path, labelsTr_path
    """
    dataset_id_str = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_path = os.path.join(nnunet_raw_path, dataset_id_str)
    imagesTr_path = os.path.join(dataset_path, 'imagesTr')
    imagesTs_path = os.path.join(dataset_path, 'imagesTs')
    labelsTr_path = os.path.join(dataset_path, 'labelsTr')
    return imagesTr_path, imagesTs_path, labelsTr_path

def get_all_folder_paths(data_path):
    """
    Get all patient folder paths from the given data path.

    Args:
        data_path (str): The path containing all patient folders.

    Returns:
        list: A list containing the full paths of all patient folders.
    """
    # Only get immediate subdirectories
    all_folder_paths = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    return sorted(all_folder_paths)

def get_all_file_paths(folder_path, modalities_to_include=None):
    """
    Get all file paths from the given patient folder path and organize them by modality.

    Args:
        folder_path (str): The path to a patient folder.
        modalities_to_include (list, optional): List of modalities to include. If None, include all modalities.

    Returns:
        dict: A dictionary containing the modalities as keys and the corresponding file paths as values.
    """
    # Initialize an empty dictionary to store file paths by modality
    modality_file_paths = {}

    # Define all possible modalities
    all_modalities = ["seg", "t1c", "t1n", "t2f", "t2w"]

    # If modalities_to_include is None, include all modalities
    if modalities_to_include is None:
        modalities_to_include = all_modalities

    # Walk through the folder path, extracting file names
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Extract the modality from the filename (e.g., "-seg", "-t1c", etc.)
            for modality in modalities_to_include:
                if f'-{modality}.' in file:
                    # Store the full path of the file with the modality as the key
                    modality_file_paths[modality] = os.path.join(root, file)
                    break

    return modality_file_paths

def rename_single_file_for_nnunet_convention_from_string(file_path_string, num_patient, modality, dataset_name="BRATS"):
    """
    Rename a single file to follow the nnU-Net naming convention.

    Parameters:
        file_path_string (str): The original file path as a string (not used in the renaming process).
        num_patient (int): The patient number, which will be zero-padded to 4 digits.
        modality (str): The modality of the image (e.g., 't1c', 't1n', 't2f', 't2w', or 'seg').
        dataset_name (str, optional): The name of the dataset. Defaults to "BRATS".

    Returns:
        str: The new file name following nnU-Net naming convention.
    """
    # Format the patient number to always be 4 characters, zero-padded
    patient_id = f"{num_patient:04d}"

    # Define a mapping of modality to the corresponding suffix used in the nnU-Net naming convention
    modality_mapping = {
        "t1c": "0000",
        "t1n": "0001",
        "t2f": "0002",
        "t2w": "0003"
    }

    # Construct the new file name based on modality
    if modality in modality_mapping:
        new_file_name = f"{dataset_name}_{patient_id}_{modality_mapping[modality]}.nii.gz"
    elif modality == "seg":
        new_file_name = f"{dataset_name}_{patient_id}.nii.gz"
    else:
        raise ValueError("Invalid modality. Supported modalities: 't1c', 't1n', 't2f', 't2w', 'seg'")

    return new_file_name

def rename_dictionary_patient_file_paths_for_nnunet_convention(dict_file_paths, num_patient, dataset_name="BRATS"):
    """
    Renames file paths in a given dictionary of patient file paths according to the nnU-Net naming convention.

    Parameters:
        dict_file_paths (dict): A dictionary where keys are modality names and values are file paths.
        num_patient (int): The patient number to be used in the renamed file paths.
        dataset_name (str, optional): The name of the dataset, default is 'BRATS'.

    Returns:
        dict: A dictionary with modalities as keys and new file names as values.
    """
    res = {}

    for modality, file_path in dict_file_paths.items():
        res[modality] = rename_single_file_for_nnunet_convention_from_string(
            file_path_string=file_path, num_patient=num_patient, modality=modality, dataset_name=dataset_name)
    return res

def create_dataset_json(dataset_path, modalities, numTraining, file_ending=".nii.gz"):
    """
    Create the dataset.json file in the dataset_path directory.

    Args:
        dataset_path (str): The path to the dataset directory.
        modalities (list): List of modalities used.
        numTraining (int): Number of training samples.
        file_ending (str): File extension of the images.

    """
    # Remove 'seg' from modalities if present
    modalities = [modality for modality in modalities if modality != 'seg']

    # Define modality indices
    modality_indices = {
        "t1c": 0,
        "t1n": 1,
        "t2f": 2,
        "t2w": 3
    }

    # Create channel_names mapping using modality indices
    channel_names = {}
    for modality in modalities:
        if modality in modality_indices:
            index = modality_indices[modality]
            channel_names[str(index)] = modality.upper()
        else:
            raise ValueError(f"Unknown modality '{modality}'. Supported modalities are: {list(modality_indices.keys())}")

    # Fixed labels
    labels = {
        "background": 0,
        "NCR": 1,
        "ED": 2,
        "ET": 3
    }

    dataset_json = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": numTraining,
        "file_ending": file_ending
    }

    dataset_json_path = os.path.join(dataset_path, "dataset.json")
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)


def process_brats2023_for_nnunet(brats_path, nnunet_raw_path, dataset_id, dataset_name='brats', modalities=None, sample=None, verbose=True):
    """
    Process the BraTS2023 data for nnU-Net.

    Args:
        brats_path (str): The path to the data containing all patient folders.
        nnunet_raw_path (str): The path to the nnUNet_raw directory.
        dataset_id (int): The dataset ID (an integer between 1 and 999).
        dataset_name (str, optional): The dataset name. Defaults to 'brats'.
        modalities (list, optional): List of modalities to include. If None, include all.
        sample (int, optional): For testing, use only the first 'sample' patients. Defaults to None.
        verbose (bool, optional): Verbose output. Defaults to True.

    Returns:
        list: List of file paths that could not be processed due to exceptions.
    """
    list_of_path_exceptions = []

    # Get the nnU-Net dataset paths
    imagesTr_path, imagesTs_path, labelsTr_path = get_nnunet_dataset_paths(nnunet_raw_path, dataset_id, dataset_name)

    # Delete all files in imagesTr_path, imagesTs_path, labelsTr_path
    for path in [imagesTr_path, imagesTs_path, labelsTr_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # Also delete the dataset.json file if it exists
    dataset_json_path = os.path.join(os.path.dirname(imagesTr_path), "dataset.json")
    if os.path.exists(dataset_json_path):
        os.remove(dataset_json_path)
        

    # Get all patient folders
    patient_folders = get_all_folder_paths(brats_path)

    # For testing, use only the first "sample" elements of the patient_folders list
    if sample is not None:
        patient_folders = patient_folders[:sample]

    if verbose:
        print()
        print("--------------------------------------------------")
        print(f"Processing {len(patient_folders)} patient folders...")
        print()

    num_patient = 0  # Initialize the patient counter

    # Process each patient folder using tqdm for progress visualization
    for idx, patient_folder in enumerate(tqdm(patient_folders, desc="Processing patients")):

        # Get all file paths for the current patient folder
        dict_file_paths = get_all_file_paths(patient_folder, modalities_to_include=modalities)

        if not dict_file_paths:
            if verbose:
                print(f"No valid modalities found in {patient_folder}. Skipping.")
            continue

        # Rename the file paths for the current patient folder
        renamed_file_paths = rename_dictionary_patient_file_paths_for_nnunet_convention(
            dict_file_paths, num_patient, dataset_name=dataset_name
        )

        # Try to load and save each image using nibabel, and collect exceptions
        failed_to_process = False
        for modality, file_path in dict_file_paths.items():
            try:
                # Load the image
                img = nib.load(file_path)

                # Determine the new file path
                if modality != "seg":
                    new_file_path = os.path.join(imagesTr_path, renamed_file_paths[modality])
                elif modality == "seg":
                    new_file_path = os.path.join(labelsTr_path, renamed_file_paths[modality])
                else:
                    raise ValueError("Invalid modality. Supported modalities: 't1c', 't1n', 't2f', 't2w', 'seg'")

                # Save the image to the new file path
                nib.save(img, new_file_path)

            except Exception as e:
                # Print the original path when an exception occurs
                print(f"Failed to process {file_path}: {e}")
                list_of_path_exceptions.append(file_path)
                failed_to_process = True
                break  # Stop processing this patient

        if failed_to_process:
            if verbose:
                print(f"Skipping patient {num_patient} due to errors.")
            continue  # Skip to next patient

        num_patient += 1  # Increment the patient counter after processing

    print("There are", num_patient, "patients in the dataset.")

    # Get the number of training samples (number of label files)
    numTraining = len(os.listdir(labelsTr_path))

    numImages = len(os.listdir(imagesTr_path))

    # The number of training images should be divisible by the number of labels

    if numImages % numTraining != 0:
        raise ValueError(f"Number of images ({numImages}) is not divisible by the number of labels ({numTraining}).")


    # Get the dataset path
    dataset_path = os.path.dirname(imagesTr_path)

    # Create dataset.json file
    create_dataset_json(dataset_path, modalities, numTraining, file_ending=".nii.gz")

    if verbose:
        print(f"dataset.json has been created at {dataset_path}")

    return list_of_path_exceptions

def main():
    parser = argparse.ArgumentParser(description="Process BraTS2023 data for nnU-Net.")

    parser.add_argument('--brats_path', required=True, type=str, help='The path to the data containing all patient folders.')
    parser.add_argument('--nnunet_raw_path', required=True, type=str, help='The path to the nnUNet_raw directory.')
    parser.add_argument('--dataset_id', required=True, type=int, help='The dataset ID (an integer between 1 and 999).')
    parser.add_argument('--dataset_name', required=False, default='brats', type=str, help='The dataset name (e.g., brats).')
    parser.add_argument('--modalities', nargs='+', required=False, default=None, help='List of modalities to include, e.g., t1c t1n t2f t2w seg.')
    parser.add_argument('--sample', type=int, default=None, help='For testing, use only the first N patients.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

    args = parser.parse_args()

    process_brats2023_for_nnunet(
        brats_path=args.brats_path,
        nnunet_raw_path=args.nnunet_raw_path,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        modalities=args.modalities,
        sample=args.sample,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
