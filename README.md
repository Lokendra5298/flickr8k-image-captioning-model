# Flickr8k Image Captioning

This project implements an image captioning model on the Flickr8k dataset using PyTorch.

## Contents
- Download the Dataset
- Dataset Path Configuration
- Model Overview
  - CNN Encoder
  - LSTM Decoder
- Training
- Checkpoints & Logs
- Plot Loss and Accuracy Curves
- Generate Caption for a Single Image
- Requirements
- Contributing
- License

## Download the Dataset

Run the dataset download script:

```bash
python src/download_dataset.py
```

After download, the dataset should look like this:

```
data/flickr8k/flickr8k/
├── Images/
└── captions.txt
```

You can verify the dataset using:

```bash
find data/flickr8k -type f -name "*.jpg" | head
```

and:

```bash
find data/flickr8k -type f -name "captions.txt"
```

### Dataset Path Configuration

By default, the code searches inside:

```
data/flickr8k
```

If your dataset is somewhere else, set the environment variable:

```bash
export FLICKR8K_DIR=/path/to/your/flickr8k
```

## Model Overview

1. CNN Encoder

The encoder uses a pretrained ResNet18 model. The final classification layer is removed, and the image is converted into a fixed-size feature vector.

```
Image -> ResNet18 -> Feature vector
```

2. LSTM Decoder

The decoder takes the image feature vector and generates a caption word by word.

```
Image feature + previous words -> LSTM -> next word prediction
```

## Training

To train the model, run:

```bash
python src/train.py
```

- The checkpoint is saved at:

```
checkpoints/caption_model.pth
```

- The training log is saved at:

```
logs/training_log.csv
```

## Plot Loss and Accuracy Curves

After training, generate plots using:

```bash
python src/plot_metrics.py
```

This creates:

```
plots/loss_curve.png
plots/accuracy_curve.png
```

## Generate Caption for a Single Image

Use:

```bash
python src/predict.py --image data/flickr8k/flickr8k/Images/1356796100_b265479721.jpg
```

You can also pass only the image filename if `predict.py` supports automatic image search.

Example generated caption:

```
a man is standing in the snow
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib

Install dependencies (recommended in a virtual environment):

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, install the primary dependencies manually.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with a clear description of your change.

## License

This project does not include a license file. Add a LICENSE if you want to define reuse terms.
