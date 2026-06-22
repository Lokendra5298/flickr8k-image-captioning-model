# src/plot_metrics.py

import os
import pandas as pd
import matplotlib.pyplot as plt

from config import CFG


def plot_available_columns(df, columns, title, ylabel, output_path, multiply_by_100=False):
    os.makedirs(CFG.PLOT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plotted_anything = False

    for column, label in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")

            if values.notna().sum() == 0:
                continue

            if multiply_by_100:
                values = values * 100

            plt.plot(df["epoch"], values, marker="o", label=label)
            plotted_anything = True

    if not plotted_anything:
        print(f"Skipping plot: {title}. No valid columns found.")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_path}")


def main():
    if not os.path.exists(CFG.TRAIN_LOG_PATH):
        raise FileNotFoundError(
            f"No training log found at: {CFG.TRAIN_LOG_PATH}\n"
            f"Run training first using: python src/train.py"
        )

    df = pd.read_csv(CFG.TRAIN_LOG_PATH)

    if "epoch" not in df.columns:
        raise ValueError("training_log.csv must contain an 'epoch' column.")

    print("\nTraining log:")
    print(df)

    plot_available_columns(
        df=df,
        columns=[
            ("train_loss", "Train Loss"),
            ("val_loss", "Validation Loss"),
            ("test_loss", "Test Loss"),
        ],
        title="Training, Validation, and Test Loss",
        ylabel="Loss",
        output_path=CFG.LOSS_PLOT_PATH,
        multiply_by_100=False
    )

    plot_available_columns(
        df=df,
        columns=[
            ("train_acc", "Train Accuracy"),
            ("val_acc", "Validation Accuracy"),
            ("test_acc", "Test Accuracy"),
        ],
        title="Training, Validation, and Test Token Accuracy",
        ylabel="Token Accuracy (%)",
        output_path=CFG.ACC_PLOT_PATH,
        multiply_by_100=True
    )


if __name__ == "__main__":
    main()