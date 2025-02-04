import os
import shutil
import nibabel as nib
import re
import argparse
from tqdm import tqdm
import json
import numpy as np

class LabelEncoder:
    def __init__(self, list_of_classes):
        self.list_of_classes = list_of_classes
        self.class_to_index = {class_: i for i, class_ in enumerate(list_of_classes)}
        self.index_to_class = {i: class_ for i, class_ in enumerate(list_of_classes)}

    def encode(self, class_):
        return self.class_to_index[class_]
    
    def decode(self, index):
        return self.index_to_class[index]
    
    def __len__(self):
        return len(self.list_of_classes)
    
    def __repr__(self):
        return f"LabelEncoder({self.list_of_classes})"
    
    def __str__(self):
        return f"LabelEncoder with classes: {self.list_of_classes}"
    
# instantiate the label encoder
list_of_ground_truth_classes = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 29, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 72, 77, 80, 85, 501, 502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 520, 530]
label_encoder = LabelEncoder(list_of_ground_truth_classes)
    
def encode_image(img_data, label_encoder):
    # Apply the label encoder directly using numpy's vectorize
    vectorized_encode = np.vectorize(label_encoder.encode)
    encoded_img_data = vectorized_encode(img_data)
    return encoded_img_data

import os
import json

def create_dataset_json(dataset_path, modalities, numTraining, label_encoder, file_ending=".nii.gz"):
    """
    Create the dataset.json file in the dataset_path directory.

    Args:
        dataset_path (str): The path to the dataset directory.
        modalities (list): List of modalities used.
        numTraining (int): Number of training samples.
        label_encoder (LabelEncoder): An instance of the LabelEncoder.
        file_ending (str): File extension of the images.
    """
    # Define the fixed modality indices
    modality_indices = {
        "T1": 1,
        "T2": 3
    }

    # Remove 'seg' from modalities if present
    modalities_no_seg = [modality for modality in modalities if modality != 'seg']

    # Assign indices to modalities based on the fixed mapping
    channel_names = {
        f"{modality_indices[modality]}": modality.upper() 
        for modality in modalities_no_seg if modality in modality_indices
    }

    # Original labels mapping from class names to original class indices
    original_labels = {
        "background": 0,
        "left_cerebral_white_matter": 2,
        "left_cerebral_cortex": 3,
        "left_lateral_ventricle": 4,
        "left_inferior_lateral_ventricle": 5,
        "left_cerebellum_white_matter": 7,
        "left_cerebellum_cortex": 8,
        "left_thalamus_proper": 10,
        "left_caudate": 11,
        "left_putamen": 12,
        "left_pallidum": 13,
        "third_ventricle": 14,
        "fourth_ventricle": 15,
        "brain_stem": 16,
        "left_hippocampus": 17,
        "left_amygdala": 18,
        "csf": 24,
        "left_accumbens_area": 26,
        "left_ventral_dc": 28,
        "left_undetermined": 29,
        "left_vessel": 30,
        "left_choroid_plexus": 31,
        "right_cerebral_white_matter": 41,
        "right_cerebral_cortex": 42,
        "right_lateral_ventricle": 43,
        "right_inferior_lateral_ventricle": 44,
        "right_cerebellum_white_matter": 46,
        "right_cerebellum_cortex": 47,
        "right_thalamus_proper": 49,
        "right_caudate": 50,
        "right_putamen": 51,
        "right_pallidum": 52,
        "right_hippocampus": 53,
        "right_amygdala": 54,
        "right_accumbens_area": 58,
        "right_ventral_dc": 60,
        "right_vessel": 62,
        "right_choroid_plexus": 63,
        "fifth_ventricle": 72,
        "wm_hypointensities": 77,
        "non_wm_hypointensities": 80,
        "optic_chiasm": 85,
        "air_internal": 501,
        "artery": 502,
        "eyes": 506,
        "other_tissues": 507,
        "rectus_muscles": 508,
        "mucosa": 509,
        "skin": 511,
        "spinal_cord": 512,
        "vein": 514,
        "bone_cortical": 515,
        "bone_cancellous": 516,
        "cortical_csf": 520,
        "optic_nerve": 530
    }

    # Create new labels mapping where keys are class names and values are new class indices
    labels = {}
    for class_name, original_class_index in original_labels.items():
        if original_class_index in label_encoder.class_to_index:
            new_class_index = label_encoder.encode(original_class_index)
            labels[class_name] = new_class_index  # Correct format: {class_name: integer}

    dataset_json = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": numTraining,
        "file_ending": file_ending
    }

    dataset_json_path = os.path.join(dataset_path, "dataset.json")
    
    # Check if the JSON file exists, print a message, and delete it if it does
    if os.path.exists(dataset_json_path):
        print(f"dataset.json already exists at {dataset_json_path}. Deleting it.")
        os.remove(dataset_json_path)

    # Write the new JSON file
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)




def get_all_ukb_identifiers(folder_path):
    """
    Get all UKB identifiers from the given folder path.
    
    Args:
        folder_path (str): Path to the folder containing patient files.
        
    Returns:
        list: Sorted list of unique UKB identifiers.
    """
    files = os.listdir(folder_path)
    patient_ids = set()
    for filename in files:
        if filename.startswith('UKB'):
            match = re.match(r'(UKB\d+)', filename)
            if match:
                patient_ids.add(match.group(1))
    return sorted(patient_ids)

def create_file_paths_from_ukb_identifier(ukb_identifier, folder_path, modalities):
    """
    Create file paths for modalities for a given UKB identifier.

    Args:
        ukb_identifier (str): UKB identifier for the patient.
        folder_path (str): Path to the folder containing patient files.
        modalities (list): List of modalities.

    Returns:
        dict: File paths for modalities.
    """
    file_paths = {}
    for modality in modalities:
        if modality == 'seg':
            file_name = f"{ukb_identifier}_labeling.nii.gz"
        else:
            modality_upper = modality.upper()
            file_name = f"{ukb_identifier}_{modality_upper}.nii.gz"
        file_path = os.path.join(folder_path, file_name)
        file_paths[modality] = file_path
    return file_paths

def check_that_files_exist(file_paths):
    """
    Check if all files in the dictionary exist.

    Args:
        file_paths (dict): Dictionary of file paths.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    return all(os.path.exists(path) for path in file_paths.values())

def rename_single_file_for_nnunet_convention_from_string(file_path_string, num_patient, modality, imagesTr_path, labelsTr_path, modality_indices, dataset_name="synthseg"):
    """
    Rename a single file to follow the nnU-Net naming convention.

    Args:
        file_path_string (str): The original file path as a string (not used in the renaming process).
        num_patient (int): The patient number, which will be zero-padded to 4 digits.
        modality (str): The modality of the image (e.g., 'T1', 'T2', 'seg').
        imagesTr_path (str): Path to the imagesTr directory.
        labelsTr_path (str): Path to the labelsTr directory.
        modality_indices (dict): Dictionary mapping modalities to indices.
        dataset_name (str, optional): The name of the dataset. Defaults to "synthseg".

    Returns:
        str: The new file name following nnU-Net naming convention.
    """
    patient_id = f"{num_patient:04d}"

    if modality != "seg":
        modality_index = modality_indices[modality]
        modality_index_padded = f"{modality_index:04d}"  # Ensure 4-digit padding for modality
        new_file_name = f"{dataset_name}_{patient_id}_{modality_index_padded}.nii.gz"
        new_file_name = os.path.join(imagesTr_path, new_file_name)
    elif modality == "seg":
        new_file_name = f"{dataset_name}_{patient_id}.nii.gz"
        new_file_name = os.path.join(labelsTr_path, new_file_name)
    else:
        raise ValueError(f"Invalid modality '{modality}'. Modality not recognized.")

    return new_file_name

def rename_dictionary_patient_file_paths_for_nnunet_convention(old_dict, num_patient, imagesTr_path, labelsTr_path, imagesTs_path, modality_indices, dataset_name="synthseg"):
    """
    Rename file paths in a given dictionary of patient file paths according to the nnU-Net naming convention.

    Args:
        old_dict (dict): Dictionary with keys as modality names and values as file paths.
        num_patient (int): The patient number to be used in the renamed file paths.
        imagesTr_path (str): Path to the imagesTr directory.
        labelsTr_path (str): Path to the labelsTr directory.
        modality_indices (dict): Dictionary mapping modalities to indices.
        dataset_name (str, optional): The name of the dataset. Defaults to "synthseg".

    Returns:
        dict: Dictionary with modalities as keys and new file names as values.
    """
    res = {}
    for modality, file_path in old_dict.items():
        res[modality] = rename_single_file_for_nnunet_convention_from_string(
            file_path_string=file_path,
            num_patient=num_patient,
            modality=modality,
            imagesTr_path=imagesTr_path,
            labelsTr_path=labelsTr_path,
            modality_indices=modality_indices,
            dataset_name=dataset_name
        )
    return res

def copy_old_paths_to_new_paths_from_dictionary(dict_file_paths, new_file_paths, label_encoder):
    """
    Copy files from old paths to new paths based on the provided dictionaries.
    If the modality is 'seg', encode the segmentation image using the label encoder before saving.
    
    Args:
        dict_file_paths (dict): Dictionary with original file paths.
        new_file_paths (dict): Dictionary with new file paths.
        label_encoder (LabelEncoder): An instance of the LabelEncoder to encode segmentation images.
    """
    for modality, old_path in dict_file_paths.items():
        new_path = new_file_paths[modality]
        if modality == 'seg':
            # Load the segmentation image
            img = nib.load(old_path)
            img_data = img.get_fdata()
            
            # Encode the segmentation image
            encoded_img_data = encode_image(img_data, label_encoder)
            
            # Preserve the original data type and header information
            encoded_img = nib.Nifti1Image(encoded_img_data.astype(img.get_data_dtype()), img.affine, img.header)
            
            # Save the encoded image to the new path
            nib.save(encoded_img, new_path)
        else:
            # For other modalities, copy the file directly
            shutil.copy(old_path, new_path)


def get_nnunet_dataset_paths(nnunet_raw_path, dataset_id, dataset_name):
    """
    Get paths to imagesTr, imagesTs, and labelsTr directories for the specified dataset.
    
    Args:
        nnunet_raw_path (str): Path to nnUNet_raw directory.
        dataset_id (int): Dataset ID (an integer between 1 and 999).
        dataset_name (str): Dataset name (e.g., 'synthseg').
    
    Returns:
        tuple: Paths to imagesTr, imagesTs, and labelsTr directories.
    """
    dataset_id_str = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_path = os.path.join(nnunet_raw_path, dataset_id_str)
    imagesTr_path = os.path.join(dataset_path, 'imagesTr')
    imagesTs_path = os.path.join(dataset_path, 'imagesTs')
    labelsTr_path = os.path.join(dataset_path, 'labelsTr')
    return imagesTr_path, imagesTs_path, labelsTr_path

def remove_all_files_in_directory(directory):
    """
    Remove all files in the specified directory.
    
    Args:
        directory (str): Path to the directory.
    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def process_ukb_dataset(synthseg_path, nnunet_raw_path, dataset_id, verbose, modalities, label_encoder, sample=None, dataset_name="synthseg"):
    """
    Process the UKB dataset, rename files according to nnU-Net convention, and copy them to the appropriate directories.

    Args:
        synthseg_path (str): Path to the folder containing SynthSeg data.
        nnunet_raw_path (str): Path to the nnUNet_raw directory.
        dataset_id (int): Dataset ID for the nnU-Net dataset.
        verbose (bool): Whether to print verbose output.
        modalities (list): List of modalities to include, e.g., ['T1', 'T2', 'seg'].
        sample (int, optional): If provided, limits the number of patients processed to the first 'sample' patients.
        dataset_name (str, optional): Name of the dataset. Defaults to "synthseg".
    """
    if verbose:
        print(f"Processing UKB dataset for dataset ID {dataset_id} and dataset name {dataset_name}...")

    imagesTr_path, imagesTs_path, labelsTr_path = get_nnunet_dataset_paths(nnunet_raw_path, dataset_id, dataset_name)

    # ig the repositories are full, print a message that they are full and you are going to delete the files in them
    if len(os.listdir(imagesTr_path)) > 0:
        print(f"imagesTr directory at {imagesTr_path} is not empty. Deleting all files.")
    if len(os.listdir(imagesTs_path)) > 0:
        print(f"imagesTs directory at {imagesTs_path} is not empty. Deleting all files.")
    if len(os.listdir(labelsTr_path)) > 0:
        print(f"labelsTr directory at {labelsTr_path} is not empty. Deleting all files.")

    



    # Create directories if they do not exist
    os.makedirs(imagesTr_path, exist_ok=True)
    os.makedirs(imagesTs_path, exist_ok=True)
    os.makedirs(labelsTr_path, exist_ok=True)

    # Remove all files in the directories
    remove_all_files_in_directory(imagesTr_path)
    remove_all_files_in_directory(imagesTs_path)
    remove_all_files_in_directory(labelsTr_path)

    # if a dataset.json file already exists, delete it
    dataset_json_path = os.path.join(os.path.dirname(imagesTr_path), "dataset.json")
    if os.path.exists(dataset_json_path):
        print(f"dataset.json already exists at {dataset_json_path}. Deleting it.")
        os.remove(dataset_json_path)

    if verbose:
        print(f"imagesTr path: {imagesTr_path}")
        print(f"imagesTs path: {imagesTs_path}")
        print(f"labelsTr path: {labelsTr_path}")

    ukb_identifiers = get_all_ukb_identifiers(synthseg_path)
    if verbose:
        print(f"Found {len(ukb_identifiers)} unique UKB identifiers.")

    # Limit to the first 'sample' patients if 'sample' is specified
    if sample is not None:
        ukb_identifiers = ukb_identifiers[:sample]
        if verbose:
            print(f"Processing only the first {sample} patients.")

    # Create modality indices mapping
    modalities_no_seg = [modality for modality in modalities if modality != 'seg']
    modality_indices = {
        "T1": 1,
        "T2": 3
    }

    num_patient = 0
    for ukb_identifier in tqdm(ukb_identifiers, desc="Processing UKB identifiers"):
        if verbose:
            print(f"Processing UKB identifier: {ukb_identifier}, Patient number: {num_patient}")

        # Create file paths based on modalities
        file_paths = create_file_paths_from_ukb_identifier(ukb_identifier, synthseg_path, modalities)

        # Check that all required files exist
        if check_that_files_exist(file_paths):
            try:
                # Attempt to load the files to ensure they are not corrupted
                for modality, file_path in file_paths.items():
                    nib.load(file_path)

                # Prepare the new file paths
                new_file_paths = rename_dictionary_patient_file_paths_for_nnunet_convention(
                    old_dict=file_paths,
                    num_patient=num_patient,
                    imagesTr_path=imagesTr_path,
                    labelsTr_path=labelsTr_path,
                    imagesTs_path=imagesTs_path,
                    modality_indices=modality_indices,
                    dataset_name=dataset_name
                )

                # Copy files
                copy_old_paths_to_new_paths_from_dictionary(file_paths, new_file_paths, label_encoder=label_encoder)
                num_patient += 1

            except Exception as e:
                print(f"Error loading files for {ukb_identifier}: {e}")
                continue
        else:
            if verbose:
                print(f"Missing files for {ukb_identifier}, skipping.")

    # Get the dataset path
    dataset_path = os.path.dirname(imagesTr_path)

    numTraining = num_patient

    # Create dataset.json file
    create_dataset_json(dataset_path, modalities, numTraining, label_encoder=label_encoder, file_ending=".nii.gz")


    if verbose:
        print(f"dataset.json has been created at {dataset_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process UKB dataset and rename files for nnU-Net.")
    parser.add_argument("--synthseg_path", type=str, required=True, help="Path to the SynthSeg training labels directory.")
    parser.add_argument("--nnunet_raw_path", type=str, required=True, help="Path to the nnUNet_raw directory.")
    parser.add_argument("--dataset_id", type=int, required=True, help="Dataset ID for the nnU-Net dataset.")
    parser.add_argument("--dataset_name", type=str, default="synthseg", help="Name of the dataset. Defaults to 'synthseg'.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument('--modalities', nargs='+', required=True, help='List of modalities to include, e.g., T1 T2 seg.')
    parser.add_argument('--sample', type=int, default=None, help="If specified, process only the first 'sample' patients.")

    args = parser.parse_args()
    process_ukb_dataset(
        synthseg_path=args.synthseg_path,
        nnunet_raw_path=args.nnunet_raw_path,
        dataset_id=args.dataset_id,
        verbose=args.verbose,
        modalities=args.modalities,
        sample=args.sample,
        dataset_name=args.dataset_name,
        label_encoder=label_encoder
    )
