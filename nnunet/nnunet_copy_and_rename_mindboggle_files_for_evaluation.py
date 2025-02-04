import os
import shutil
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Copy and rename nii.gz files to target directories.")
    parser.add_argument(
        '--modality',
        type=str,
        required=True,
        choices=['t1', 't2'],
        help='Modality type (e.g., t1, t2)'
    )
    return parser.parse_args()

def clear_target_directories(target_dirs):
    """
    Removes all .nii.gz files from the specified target directories.
    """
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            print(f"Target directory '{target_dir}' does not exist. Creating it.")
            os.makedirs(target_dir, exist_ok=True)
            continue  # No files to delete if directory was just created

        # List all .nii.gz files in the directory
        files = [f for f in os.listdir(target_dir) if f.endswith(".nii.gz")]
        for file in files:
            file_path = os.path.join(target_dir, file)
            try:
                os.remove(file_path)
                # Optionally, print or log the deletion
                # print(f"Deleted '{file_path}'")
            except Exception as e:
                print(f"Failed to delete '{file_path}': {e}")

def main():
    args = parse_arguments()

    # Define paths
    path_of_inputs = "/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/data/mindboggle/input"
    
    path_to_paste = [
        "/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_raw/Dataset022_bobs/imagesTs",
        "/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_raw/Dataset024_bobs/imagesTs",
        "/home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet/nnUNet_raw/Dataset027_bobs/imagesTs"
    ]
    
    # Modality mapping
    modality_dict = {
        "t1": "0000",
        "t2": "0001",
    }
    
    # Get modality code
    modality = args.modality.lower()
    if modality not in modality_dict:
        print(f"Error: Modality '{modality}' is not supported.")
        print(f"Supported modalities: {', '.join(modality_dict.keys())}")
        return
    modality_code = modality_dict[modality]
    
    # Get sorted list of .nii.gz files
    try:
        files = sorted([f for f in os.listdir(path_of_inputs) if f.endswith(".nii.gz")])
    except FileNotFoundError:
        print(f"Error: The input path '{path_of_inputs}' does not exist.")
        return
    except Exception as e:
        print(f"An error occurred while accessing the input directory: {e}")
        return
    
    num_files = len(files)
    print(f"Number of .nii.gz files found: {num_files}")
    
    if num_files == 0:
        print("No files to copy. Exiting.")
        return
    
    # Ensure target directories exist and clear them
    clear_target_directories(path_to_paste)
    print("Cleared existing files in target directories.")
    
    # Iterate over files with progress bar
    for idx, filename in enumerate(tqdm(files, desc="Copying files")):
        # Generate patient ID (zero-padded to 4 digits)
        patient_id = f"{idx:04d}"
        
        # Generate new filename
        new_filename = f"bobs_{patient_id}_{modality_code}.nii.gz"
        
        # Source file path
        src_path = os.path.join(path_of_inputs, filename)
        
        # Copy to each target directory
        for target_dir in path_to_paste:
            dest_path = os.path.join(target_dir, new_filename)
            try:
                shutil.copy2(src_path, dest_path)
            except Exception as e:
                print(f"Failed to copy '{src_path}' to '{dest_path}': {e}")
    
    print("File copying and renaming completed successfully.")

if __name__ == "__main__":
    main()
