#!/usr/bin/env python3

import os
import argparse
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np
import json

base_correspondance = {
    0: "Background",
    1: "Left-Cerebral-Exterior",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    6: "Left-Cerebellum-Exterior",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    77: "WM-hypointensities",
    85: "Optic-Chiasm",
    172: "Vermis"
}

symmetrical_correrspondance = {
    0: "background",
    1: "Left-Cerebral-Exterior",
    2: "Cerebral-White-Matter",
    3: "Cerebral-Cortex",
    4: "Lateral-Ventricle",
    5: "Inf-Lat-Vent",
    6: "Left-Cerebellum-Exterior",
    7: "Cerebellum-White-Matter",
    8: "Cerebellum-Cortex",
    9: "Thalamus",
    10: "Caudate",
    11: "Putamen",
    12: "Pallidum",
    13: "3rd-Ventricle",
    14: "4th-Ventricle",
    15: "Brain-Stem",
    16: "Hippocampus",
    17: "Amygdala",
    18: "CSF",
    19: "Accumbens-area",
    20: "Vessel",
    21: "Choroid-plexus",
    22: "VentralDC",
    23: "WM-hypointensities",
    24: "Optic-Chiasm",
    25: "Vermis"
}



def generate_list_augmented_triplets(path_bobs_augmented, modalities=['t1', 't2']):
    """
    Generates a list of dictionaries containing paths to T1w, T2w, and segmentation files.

    Args:
        path_bobs_augmented (str): The base directory containing 'T1w', 'T2w', and 'seg' subdirectories.
        modalities (list): List of modalities to include.

    Returns:
        list of dict: A list where each dict has keys 't1', 't2', and 'seg' pointing to respective file paths.
    """
    # Define paths to each subdirectory based on modalities
    path_bobs_seg = os.path.join(path_bobs_augmented, "seg")
    paths_modalities = {}
    if 't1' in modalities:
        paths_modalities['t1'] = os.path.join(path_bobs_augmented, "T1w")
    if 't2' in modalities:
        paths_modalities['t2'] = os.path.join(path_bobs_augmented, "T2w")

    # Check if all required directories exist
    for path in paths_modalities.values():
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")
    if not os.path.isdir(path_bobs_seg):
        raise FileNotFoundError(f"Directory not found: {path_bobs_seg}")

    # Get sorted lists of files from each directory
    list_files_seg = sorted(os.listdir(path_bobs_seg))
    list_files_modalities = {mod: sorted(os.listdir(path)) for mod, path in paths_modalities.items()}

    # Ensure all modality lists have the same number of files as seg
    for mod, files in list_files_modalities.items():
        if len(files) != len(list_files_seg):
            raise ValueError(f"The number of files in '{mod}' does not match 'seg' directory.")

    list_triplets = []

    # Iterate through the files and create triplet dictionaries
    for idx in range(len(list_files_seg)):
        triplet = {'seg': os.path.join(path_bobs_seg, list_files_seg[idx])}
        for mod in modalities:
            triplet[mod] = os.path.join(paths_modalities[mod], list_files_modalities[mod][idx])
        list_triplets.append(triplet)

    return list_triplets


def replace_seg_values(seg_data, symmetric=False):
    """
    If symmetric=True, remap the original (left/right) labels to the
    new consecutive integer labels (merging left and right).
    Otherwise, just return the data as uint8.
    """
    if not symmetric:
        # Just convert to integer type (uint8).
        return seg_data.astype(np.uint8)

    # This map merges right labels onto new consecutive integers in symmetrical_correrspondance
    symmetrical_label_map = {
        0: 0,   # Background
        1: 1,   # Left-Cerebral-Exterior
        2: 2,   41: 2,   # Cerebral-White-Matter
        3: 3,   42: 3,   # Cerebral-Cortex
        4: 4,   43: 4,   # Lateral-Ventricle
        5: 5,   44: 5,   # Inf-Lat-Vent
        6: 6,             # Left-Cerebellum-Exterior
        7: 7,   46: 7,   # Cerebellum-White-Matter
        8: 8,   47: 8,   # Cerebellum-Cortex
        10: 9, 49: 9,  # Thalamus
        11: 10, 50: 10,  # Caudate
        12: 11, 51: 11,  # Putamen
        13: 12, 52: 12,  # Pallidum
        14: 13,          # 3rd-Ventricle
        15: 14,          # 4th-Ventricle
        16: 15,          # Brain-Stem
        17: 16, 53: 16,  # Hippocampus
        18: 17, 54: 17,  # Amygdala
        24: 18,          # CSF
        26: 19, 58: 19,  # Accumbens-area
        30: 20, 62: 20,  # Vessel
        31: 21, 63: 21,  # Choroid-plexus
        28: 22, 60: 22,          # Right-VentralDC + Left-VentralDC
        77: 23,          # WM-hypointensities
        85: 24,          # Optic-Chiasm
        172: 25          # Vermis
    }

    merged_seg = np.zeros_like(seg_data, dtype=np.uint8)
    for old_val, new_val in symmetrical_label_map.items():
        merged_seg[seg_data == old_val] = new_val

    return merged_seg







# ------------------------------------------------------------------
# Import or paste in your triplet functions here:
# ------------------------------------------------------------------
# Example placeholders (assume they're in the same directory):
# from your_triplet_module import get_list_triplets

# Update the get_list_triplets function to accept modalities:

def get_list_triplets(normal_path, modalities=['t1', 't2']):
    """
    ... (existing docstring)
    
    Args:
        ... (existing args)
        modalities (list): List of modalities to include.
    """
    import os

    triplet_dict = {}
    
    for root, _, files in os.walk(normal_path):
        for file in files:
            if file.endswith(".nii.gz"):
                if root not in triplet_dict:
                    triplet_dict[root] = {"seg": None}
                    for mod in modalities:
                        triplet_dict[root][mod] = None
                
                full_path = os.path.join(root, file)
                if "T1w" in file and 't1' in modalities:
                    triplet_dict[root]["t1"] = full_path
                elif "T2w" in file and 't2' in modalities:
                    triplet_dict[root]["t2"] = full_path
                elif "seg" in file:
                    triplet_dict[root]["seg"] = full_path

    final_triplets = []

    for root, data in triplet_dict.items():
        # Check if all required modalities and seg are present
        required = all(data[mod] for mod in modalities) and data["seg"]
        if required:
            final_triplets.append(data)
    return final_triplets



# ------------------------------------------------------------------
# Helper functions adapted from the BraTS script, specialized for BOBs
# ------------------------------------------------------------------

def get_nnunet_dataset_paths(nnunet_raw_path, dataset_id, dataset_name):
    """
    Given nnunet_raw_path, dataset_id, and dataset_name, returns the paths to
    imagesTr, imagesTs, and labelsTr directories, e.g.:
      nnUNet_raw/
          DatasetXYZ_datasetName/
             imagesTr/
             imagesTs/
             labelsTr/
    """
    dataset_id_str = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_path = os.path.join(nnunet_raw_path, dataset_id_str)
    imagesTr_path = os.path.join(dataset_path, 'imagesTr')
    imagesTs_path = os.path.join(dataset_path, 'imagesTs')
    labelsTr_path = os.path.join(dataset_path, 'labelsTr')
    return imagesTr_path, imagesTs_path, labelsTr_path


def rename_bobs_file_paths_for_nnunet_convention(triplet, patient_id, dataset_name="BOBs", modalities=['t1', 't2']):
    """
    Given a single 'triplet' dict with keys 't1', 't2', 'seg', rename them
    for nnU-Net's naming convention.

    e.g. BOBs_0000_0000.nii.gz for T1,
         BOBs_0000_0001.nii.gz for T2,
         BOBs_0000.nii.gz      for seg
    """
    # Zero-padded 4-digit patient ID
    pid_str = f"{patient_id:04d}"

    # We'll hard-code the mapping:
    # T1W -> channel 0000
    # T2W -> channel 0001
    # seg -> no channel index in the filename
    renamed = {}

    if 't1' in modalities:
        renamed['t1'] = f"{dataset_name}_{pid_str}_0000.nii.gz"  # t1 is always 0000
    if 't2' in modalities:
        renamed['t2'] = f"{dataset_name}_{pid_str}_0001.nii.gz"  # t2 is always 0001
    renamed['seg'] = f"{dataset_name}_{pid_str}.nii.gz"

    return renamed


def create_dataset_json(dataset_path, 
                        num_training, 
                        file_ending=".nii.gz",
                        symmetric=False,
                        modalities=['t1', 't2']):
    """
    Create a dataset.json that uses the base correspondence if symmetric=False,
    or the symmetrical_correrspondance (with consecutive integers) if symmetric=True.
    BUT now we want the final JSON to have keys=label_names, values=label_ids.
    modalities (list): List of modalities to include.
    """
    # 1) Pick the correct dictionary
     # 1) Pick the correct dictionary
    if symmetric:
        dict_to_invert = symmetrical_correrspondance
    else:
        dict_to_invert = base_correspondance

    # 2) Invert to {label_name: label_id}
    label_dict = {label_name: int(label_id) for label_id, label_name in dict_to_invert.items()}

    # 3) Define channel_names based on selected modalities with hardcoded indices
    channel_names = {}
    if 't1' in modalities:
        channel_names["0"] = "T1W"
    if 't2' in modalities:
        channel_names["1"] = "T2W"

    dataset_json = {
        "channel_names": channel_names,
        "labels": label_dict,
        "numTraining": num_training,
        "file_ending": file_ending
    }

    dataset_json_path = os.path.join(dataset_path, "dataset.json")
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)




def process_bobs_for_nnunet(bobs_path,
                            nnunet_raw_path,
                            dataset_id,
                            dataset_name='BOBs',
                            bobs_augmented_path=None,  # Added parameter
                            sample=None,
                            verbose=True,
                            symmetric=False,
                            modalities=['t1', 't2']):
    """
    Process the BOBs data for nnU-Net by:
      1) Getting triplets (T1, T2, seg) from normal and augmented paths.
      2) Creating the standard nnU-Net folder structure (imagesTr, imagesTs, labelsTr).
      3) Copying / renaming them according to nnU-Net conventions.
      4) Creating a dataset.json file describing the dataset.

    Args:
        bobs_path (str): Path containing the BOBs normal data (where T1, T2, seg are located).
        nnunet_raw_path (str): Path to the nnUNet_raw directory.
        dataset_id (int): Numeric ID for the dataset (1-999).
        dataset_name (str, optional): Name to embed in the dataset. Defaults to 'BOBs'.
        bobs_augmented_path (str, optional): Path containing the BOBs augmented data. Defaults to None.
        sample (int, optional): For testing, use only the first N triplets. Defaults to None.
        verbose (bool, optional): Verbose output. Defaults to True.
        symmetric (bool, optional): Whether to merge symmetric classes. Defaults to False.
        modalities (list): List of modalities to include, e.g., ['t1', 't2'], ['t1'], or ['t2'].

    Returns:
        list: A list of file paths that could not be processed due to exceptions.
    """
    # 1) Get triplets from normal path
    triplets = get_list_triplets(bobs_path, modalities=modalities)
    if sample is not None:
        triplets = triplets[:sample]

    if verbose:
        print(f"Found {len(triplets)} complete triplets in '{bobs_path}'.")

    # 1a) If augmented path is provided, get triplets from augmented path
    if bobs_augmented_path:
        augmented_triplets = generate_list_augmented_triplets(bobs_augmented_path, modalities=modalities)
        if sample is not None:
            augmented_triplets = augmented_triplets[:sample]
        triplets.extend(augmented_triplets)
        if verbose:
            print(f"Found {len(augmented_triplets)} complete augmented triplets in '{bobs_augmented_path}'.")

    # 2) Create / clean nnU-Net folder structure
    imagesTr_path, imagesTs_path, labelsTr_path = get_nnunet_dataset_paths(
        nnunet_raw_path, dataset_id, dataset_name
    )

    for path in [imagesTr_path, imagesTs_path, labelsTr_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # Also remove existing dataset.json if present
    dataset_json_path = os.path.join(os.path.dirname(imagesTr_path), "dataset.json")
    if os.path.exists(dataset_json_path):
        os.remove(dataset_json_path)

    # 3) Copy / rename each triplet
    list_of_exceptions = []
    patient_id = 0

# In the process_bobs_for_nnunet function, update the copying section:

    for i, triplet in enumerate(tqdm(triplets, desc="Copying BOBs triplets")):
        renamed = rename_bobs_file_paths_for_nnunet_convention(
            triplet, patient_id, dataset_name=dataset_name, modalities=modalities  # Pass modalities
        )

        try:
            # Process T1 if included
            if 't1' in modalities:
                img_t1 = nib.load(triplet['t1'])
                outpath_t1 = os.path.join(imagesTr_path, renamed['t1'])
                nib.save(img_t1, outpath_t1)

            # Process T2 if included
            if 't2' in modalities:
                img_t2 = nib.load(triplet['t2'])
                outpath_t2 = os.path.join(imagesTr_path, renamed['t2'])
                nib.save(img_t2, outpath_t2)

            # Always process segmentation
            img_seg = nib.load(triplet['seg'])
            seg_data = img_seg.get_fdata().astype(np.uint8)
            seg_data = replace_seg_values(seg_data, symmetric=symmetric)
            img_seg_uint8 = nib.Nifti1Image(seg_data, img_seg.affine, img_seg.header)

            outpath_seg = os.path.join(labelsTr_path, renamed['seg'])
            nib.save(img_seg_uint8, outpath_seg)

        except Exception as e:
            list_of_exceptions.append(triplet)
            if verbose:
                print(f"Failed to process triplet {triplet} due to {e}")
            continue

        patient_id += 1


    # 4) Create dataset.json
    # Our training samples are those that have segmentation
    # i.e., number of label files
    num_training = len(os.listdir(labelsTr_path))
    dataset_path = os.path.dirname(imagesTr_path)

    create_dataset_json(dataset_path, num_training=num_training, file_ending=".nii.gz", symmetric=symmetric, modalities=modalities)

    if verbose:
        print(f"dataset.json created at {dataset_path}")
        print(f"Processed {patient_id} triplets successfully.")
        if list_of_exceptions:
            print(f"{len(list_of_exceptions)} triplets failed. See return value for details.")

    return list_of_exceptions



def main():
    parser = argparse.ArgumentParser(description="Process BOBs data for nnU-Net.")
    parser.add_argument('--bobs_path', required=True, type=str,
                        help='Path to the BOBs data (where T1, T2, seg are located).')
    parser.add_argument('--bobs_augmented_path', required=False, type=str,
                        help='Path to the BOBs augmented data (where T1, T2, seg are located).')

    parser.add_argument('--nnunet_raw_path', required=True, type=str,
                        help='Path to the nnUNet_raw directory.')
    parser.add_argument('--dataset_id', required=True, type=int,
                        help='The dataset ID (an integer between 1 and 999).')
    parser.add_argument('--dataset_name', required=False, default='BOBs', type=str,
                        help='The dataset name (e.g., BOBs).')
    parser.add_argument('--sample', type=int, default=None,
                        help='For testing, use only the first N triplets.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output.')
    parser.add_argument('--symmetric', action='store_true',
                        help='Merge symmetric classes.')
    # In the main() function, within the argparse section, add:

    parser.add_argument('--modalities', required=True, nargs='+', choices=['t1', 't2'],
                        help='Modalities to include, e.g., t1 t2, t1, or t2.')


    args = parser.parse_args()

    # Run the main BOBs processing function
    process_bobs_for_nnunet(
        bobs_path=args.bobs_path,
        bobs_augmented_path=args.bobs_augmented_path,
        nnunet_raw_path=args.nnunet_raw_path,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        sample=args.sample,
        verbose=args.verbose,
        symmetric=args.symmetric,
        modalities=args.modalities
    )

if __name__ == "__main__":
    main()
