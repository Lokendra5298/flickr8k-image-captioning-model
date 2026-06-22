# src/config.py

import os
import torch


class Config:
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    DATA_DIR = os.environ.get(
        "FLICKR8K_DIR",
        os.path.join(PROJECT_ROOT, "data", "flickr8k")
    )

    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "caption_model.pth")

    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    TRAIN_LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")

    PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
    LOSS_PLOT_PATH = os.path.join(PLOT_DIR, "loss_curve.png")
    ACC_PLOT_PATH = os.path.join(PLOT_DIR, "accuracy_curve.png")

    IMAGE_SIZE = 224

    MIN_FREQ = 5

    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    DROPOUT = 0.3

    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 3e-4
    NUM_WORKERS = 2

    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    RANDOM_SEED = 42

    DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"


CFG = Config()