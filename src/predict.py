# src/predict.py

import argparse
import os
import torch
from PIL import Image

from config import CFG
from vocabulary import Vocabulary
from dataset import get_transforms, find_image_directory
from model import ImageCaptioningModel


def resolve_image_path(image_arg):
    """
    Allows both:
        python src/predict.py --image path/to/image.jpg

    and:
        python src/predict.py --image image_name.jpg
    """
    if os.path.exists(image_arg):
        return image_arg

    # If user only gives image filename, search inside Flickr8k image directory
    image_dir = find_image_directory(CFG.DATA_DIR)

    for root, dirs, files in os.walk(image_dir):
        for file_name in files:
            if file_name == image_arg:
                return os.path.join(root, file_name)

    raise FileNotFoundError(
        f"\nCould not find image:\n"
        f"  {image_arg}\n\n"
        f"You can pass either a full path or an image filename inside Flickr8k.\n"
    )


def load_model_and_vocab(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    vocab = Vocabulary(min_freq=checkpoint["vocab_min_freq"])
    vocab.stoi = checkpoint["vocab_stoi"]
    vocab.itos = checkpoint["vocab_itos"]

    model = ImageCaptioningModel(
        embed_size=CFG.EMBED_SIZE,
        hidden_size=CFG.HIDDEN_SIZE,
        vocab_size=len(vocab),
        num_layers=CFG.NUM_LAYERS,
        dropout=CFG.DROPOUT
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab


def generate_caption(image_path, checkpoint_path, max_length=30):
    device = torch.device(CFG.DEVICE)

    model, vocab = load_model_and_vocab(checkpoint_path, device)

    transform = get_transforms()

    resolved_image_path = resolve_image_path(image_path)

    image = Image.open(resolved_image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encoder(image)

        caption = model.decoder.generate_caption(
            image_features=image_features,
            vocab=vocab,
            max_length=max_length
        )

    return resolved_image_path, caption


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image or Flickr8k image filename"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CFG.CHECKPOINT_PATH,
        help="Path to trained checkpoint"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Maximum caption length"
    )

    args = parser.parse_args()

    image_path, caption = generate_caption(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        max_length=args.max_length
    )

    print("\nImage:")
    print(image_path)

    print("\nGenerated caption:")
    print(caption)


if __name__ == "__main__":
    main()