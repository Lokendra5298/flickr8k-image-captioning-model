# src/dataset.py

import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from vocabulary import Vocabulary
from config import CFG


def find_image_directory(search_root):
    """
    Recursively finds a folder containing Flickr8k image files.
    We look for a directory with many .jpg/.jpeg/.png files.
    """
    if not os.path.exists(search_root):
        raise FileNotFoundError(
            f"\nDataset root does not exist:\n"
            f"  {search_root}\n\n"
            f"Create this folder or set the correct path using:\n"
            f"  export FLICKR8K_DIR=/actual/path/to/flickr8k\n"
        )

    best_dir = None
    best_count = 0

    for root, dirs, files in os.walk(search_root):
        image_files = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_files) > best_count:
            best_count = len(image_files)
            best_dir = root

    if best_dir is None or best_count == 0:
        raise FileNotFoundError(
            f"\nCould not find any image folder under:\n"
            f"  {search_root}\n\n"
            f"Your Flickr8k folder should contain many .jpg files.\n"
        )

    print(f"Found image directory: {best_dir}")
    print(f"Number of image files found: {best_count}")

    return best_dir


def find_captions_file(search_root):
    """
    Recursively finds a Flickr8k captions file.
    Supports common names:
    - captions.txt
    - Flickr8k.token.txt
    """
    possible_names = {
        "captions.txt",
        "flickr8k.token.txt",
        "flickr8k.lemma.token.txt",
    }

    found_files = []

    for root, dirs, files in os.walk(search_root):
        for file_name in files:
            if file_name.lower() in possible_names:
                found_files.append(os.path.join(root, file_name))

    if len(found_files) == 0:
        raise FileNotFoundError(
            f"\nCould not find captions file under:\n"
            f"  {search_root}\n\n"
            f"Expected one of:\n"
            f"  captions.txt\n"
            f"  Flickr8k.token.txt\n"
        )

    # Prefer captions.txt if available
    for path in found_files:
        if os.path.basename(path).lower() == "captions.txt":
            print(f"Found captions file: {path}")
            return path

    print(f"Found captions file: {found_files[0]}")
    return found_files[0]


def read_captions_file(captions_file):
    """
    Supports two common Flickr8k formats.

    Format 1: captions.txt
        image,caption
        1000268201_693b08cb0e.jpg,A child in a pink dress is climbing stairs.

    Format 2: Flickr8k.token.txt
        1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing stairs.
    """
    image_to_captions = {}

    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.lower().startswith("image,caption"):
            continue

        image_name = None
        caption = None

        # Old Flickr8k.token.txt format
        if "\t" in line:
            left, caption = line.split("\t", 1)
            image_name = left.split("#")[0].strip()
            caption = caption.strip()

        # Kaggle captions.txt format
        elif "," in line:
            image_name, caption = line.split(",", 1)
            image_name = image_name.strip()
            caption = caption.strip()

        if image_name is None or caption is None:
            continue

        if image_name not in image_to_captions:
            image_to_captions[image_name] = []

        image_to_captions[image_name].append(caption)

    if len(image_to_captions) == 0:
        raise ValueError(
            f"No captions were read from {captions_file}. "
            f"Please check the file format."
        )

    return image_to_captions


def split_data_by_image(image_to_captions, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits by image name, not individual caption.

    This prevents captions from the same image appearing in both train and validation/test.
    """
    image_names = list(image_to_captions.keys())

    random.seed(seed)
    random.shuffle(image_names)

    total_images = len(image_names)

    test_size = int(total_images * test_ratio)
    val_size = int(total_images * val_ratio)

    test_images = set(image_names[:test_size])
    val_images = set(image_names[test_size:test_size + val_size])
    train_images = set(image_names[test_size + val_size:])

    return train_images, val_images, test_images


def make_pairs(image_to_captions, image_names):
    pairs = []

    for image_name in image_names:
        captions = image_to_captions[image_name]

        for caption in captions:
            pairs.append((image_name, caption))

    return pairs


class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, pairs, vocab, transform=None):
        self.image_dir = image_dir
        self.pairs = pairs
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_name, caption = self.pairs[index]

        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        caption_indices = self.vocab.numericalize(caption)
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)

        return image, caption_tensor


class CaptionCollate:
    """
    Custom collate function for padding variable-length captions.

    Batch captions may have different lengths:
        [<start>, a, dog, runs, <end>]
        [<start>, a, man, is, riding, a, horse, <end>]

    We pad them to equal length using <pad>.
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = []
        captions = []

        for image, caption in batch:
            images.append(image)
            captions.append(caption)

        images = torch.stack(images, dim=0)

        captions = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return images, captions


def get_transforms():
    return transforms.Compose([
        transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
        transforms.ToTensor(),

        # ImageNet normalization for pretrained ResNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_loaders(vocab=None):
    print(f"Searching dataset inside: {CFG.DATA_DIR}")

    image_dir = find_image_directory(CFG.DATA_DIR)
    captions_file = find_captions_file(CFG.DATA_DIR)

    image_to_captions = read_captions_file(captions_file)

    image_to_captions = {
        image_name: captions
        for image_name, captions in image_to_captions.items()
        if os.path.exists(os.path.join(image_dir, image_name))
    }

    if len(image_to_captions) == 0:
        raise FileNotFoundError(
            f"\nCaptions were loaded, but no matching image files were found.\n\n"
            f"Image directory:\n"
            f"  {image_dir}\n\n"
            f"Captions file:\n"
            f"  {captions_file}\n"
        )

    print(f"Number of captioned images found: {len(image_to_captions)}")

    train_images, val_images, test_images = split_data_by_image(
        image_to_captions,
        val_ratio=CFG.VAL_RATIO,
        test_ratio=CFG.TEST_RATIO,
        seed=CFG.RANDOM_SEED
    )

    train_pairs = make_pairs(image_to_captions, train_images)
    val_pairs = make_pairs(image_to_captions, val_images)
    test_pairs = make_pairs(image_to_captions, test_images)

    print(f"Number of training pairs: {len(train_pairs)}")
    print(f"Number of validation pairs: {len(val_pairs)}")
    print(f"Number of test pairs: {len(test_pairs)}")

    if vocab is None:
        train_captions = [caption for _, caption in train_pairs]
        vocab = Vocabulary(min_freq=CFG.MIN_FREQ)
        vocab.build_vocab(train_captions)

    transform = get_transforms()

    train_dataset = Flickr8kDataset(
        image_dir=image_dir,
        pairs=train_pairs,
        vocab=vocab,
        transform=transform
    )

    val_dataset = Flickr8kDataset(
        image_dir=image_dir,
        pairs=val_pairs,
        vocab=vocab,
        transform=transform
    )

    test_dataset = Flickr8kDataset(
        image_dir=image_dir,
        pairs=test_pairs,
        vocab=vocab,
        transform=transform
    )

    pad_idx = vocab.stoi[vocab.pad_token]
    collate_fn = CaptionCollate(pad_idx=pad_idx)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, vocab