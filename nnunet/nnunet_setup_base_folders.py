import os
import argparse

def create_base_nnunet_folders(nnunet_path):
    """
    Create the base folder structure for nnU-Net.

    Args:
        nnunet_path (str): The base path where nnUNet directory is located.

    Example:
        You can run the command: python nnunet_setup_base_folders.py /home/fp427/rds/rds-cam-segm-7tts6phZ4tw/mission/nnunet

        The resulting folder structure will be:
        nnunet/
        ├── nnUNet_raw/
        ├── nnUNet_preprocessed/
        └── nnUNet_results/
    """
    nnunet_raw_path = os.path.join(nnunet_path, 'nnUNet_raw')
    nnunet_preprocessed_path = os.path.join(nnunet_path, 'nnUNet_preprocessed')
    nnunet_results_path = os.path.join(nnunet_path, 'nnUNet_results')

    os.makedirs(nnunet_raw_path, exist_ok=True)
    os.makedirs(nnunet_preprocessed_path, exist_ok=True)
    os.makedirs(nnunet_results_path, exist_ok=True)

    print(f"Base nnU-Net folders created at: {nnunet_path}")

def main():
    parser = argparse.ArgumentParser(description="Create base nnU-Net folder structure.")
    parser.add_argument("nnunet_path", type=str, help="The base path where nnUNet directory is located.")
    args = parser.parse_args()
    
    create_base_nnunet_folders(args.nnunet_path)

if __name__ == "__main__":
    main()

