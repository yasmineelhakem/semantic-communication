import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(data, metric, ylabel, title, filename):

    plt.figure(figsize=(9, 6))

    for model_name, group in data.groupby("model"):

        group = group.sort_values("snr_db")

        plt.plot(
            group["snr_db"],
            group[metric],
            marker="o",
            linewidth=2,
            markersize=7,
            label=model_name
        )

    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    plt.xticks([-5, 0, 5, 10, 15, 20])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join("results", filename)

    plt.savefig(output_path, dpi=300)
    print(f"Graph saved: {output_path}")

    #plt.show()
    plt.close()


def main():

    csv_path = "results/semantic_evaluation.csv"

    data = pd.read_csv(csv_path)

    os.makedirs("results", exist_ok=True)

    # Graph 1: PSNR
    plot_metric(
        data,
        metric="psnr",
        ylabel="PSNR (dB)",
        title="PSNR versus SNR",
        filename="psnr_comparison.png"
    )

    # Graph 2: SSIM
    plot_metric(
        data,
        metric="ssim",
        ylabel="SSIM",
        title="SSIM versus SNR",
        filename="ssim_comparison.png"
    )

    # Graph 3: Perceptual Loss
    plot_metric(
        data,
        metric="perceptual_loss",
        ylabel="Perceptual Loss",
        title="Perceptual Loss versus SNR",
        filename="perceptual_loss_comparison.png"
    )


if __name__ == "__main__":
    main()
