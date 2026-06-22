# src/download_dataset.py

import argparse
import os
import subprocess
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"


def run_command(command):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def download_from_kaggle(data_dir):
    """
    Downloads Flickr8k from Kaggle.

    Dataset used:
        srinivasac/flickr8k-dataset

    Expected output after unzip:
        data/flickr8k/Images/
        data/flickr8k/captions.txt
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Flickr8k into: {data_dir}")

    run_command([
        "kaggle",
        "datasets",
        "download",
        "-d",
        "srinivasac/flickr8k-dataset",
        "-p",
        str(data_dir),
        "--unzip"
    ])

    print("Kaggle download completed.")


def find_files(data_dir):
    image_count = 0
    caption_files = []

    for root, dirs, files in os.walk(data_dir):
        for file_name in files:
            lower = file_name.lower()

            if lower.endswith((".jpg", ".jpeg", ".png")):
                image_count += 1

            if lower in {
                "captions.txt",
                "flickr8k.token.txt",
                "flickr8k.lemma.token.txt",
            }:
                caption_files.append(os.path.join(root, file_name))

    return image_count, caption_files


def verify_dataset(data_dir):
    image_count, caption_files = find_files(data_dir)

    print("\nDataset verification")
    print("--------------------")
    print(f"Dataset directory: {data_dir}")
    print(f"Image files found: {image_count}")

    if caption_files:
        print("Caption files found:")
        for path in caption_files:
            print(f"  - {path}")
    else:
        print("Caption files found: 0")

    if image_count == 0:
        raise FileNotFoundError(
            "No image files found. Dataset download/unzip did not complete correctly."
        )

    if len(caption_files) == 0:
        raise FileNotFoundError(
            "No caption file found. Expected captions.txt or Flickr8k.token.txt."
        )

    print("\nDataset looks ready.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Where to store the Flickr8k dataset"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    download_from_kaggle(data_dir)
    verify_dataset(data_dir)

    print("\nUse this command before training:")
    print(f"export FLICKR8K_DIR={data_dir}")
    print("python src/train.py")


if __name__ == "__main__":
    main()