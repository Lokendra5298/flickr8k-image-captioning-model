---
layout: home
title: Building an Image Captioning Model on Flickr8k
---

# Building an Image Captioning Model on Flickr8k using PyTorch

In this project, I built an image captioning model from scratch using PyTorch. The model takes an image as input and generates a natural language caption.

The goal of this project was to understand the full image captioning pipeline:

- Downloading the Flickr8k dataset
- Loading image-caption pairs
- Building a vocabulary
- Tokenizing captions
- Creating a CNN encoder
- Creating an LSTM decoder
- Training the model
- Evaluating loss and token-level accuracy
- Plotting training curves
- Generating captions for new images

---

## Model Architecture

The model uses an encoder-decoder architecture:

```text
Image -> CNN Encoder -> LSTM Decoder -> Caption
