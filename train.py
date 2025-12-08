import torch
from torch.utils.data import DataLoader, random_split
from scwfae.data.ptbxlDS import PtbxlDS
from torchinfo import summary
from typing import Tuple
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchmetrics.aggregation import MeanMetric
from torchmetrics import MetricCollection
import auraloss
from scwfae.model.ConvolutionalEcgModules import *


MAX_EPOCHS = 1000
# LEARNING_RATE = 0.0003016868024815632
LEARNING_RATE = 1e-4
BETA_ANNEALING = False


def calculate_losses(
    preds: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    mrstft = auraloss.freq.STFTLoss()
    reproduction_loss = F.mse_loss(preds, target, reduction="sum")
    # reproduction_loss = mrstft(preds.unsqueeze(1), target.unsqueeze(1))
    # reproduction_loss = F.l1_loss(preds, target, reduction="sum")
    mean_mse = F.mse_loss(preds, target, reduction="mean")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reproduction_loss + KLD * beta, reproduction_loss, KLD, mean_mse
    # return reproduction_loss, reproduction_loss, KLD, mean_mse


def save_validation_examples(
    m: torch.nn.Module, sample_batch: torch.Tensor, sample_labels: torch.Tensor
):
    m.eval()
    with torch.no_grad():
        reconstruction, _, _ = m(sample_batch.unsqueeze(1))

    plt.plot(
        range(1000),
        sample_batch[0, :].detach().cpu().numpy(),
        label="original",
    )
    plt.plot(
        range(1000),
        reconstruction[0, :].squeeze().detach().cpu().numpy(),
        label="reconstruction",
    )
    plt.savefig("./cache/sample.png")
    plt.clf()


if __name__ == "__main__":
    torch.manual_seed(42)
    ds = PtbxlDS(lowres=True)

    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])

    train_dl = DataLoader(train_ds, batch_size=32, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds), num_workers=16)

    sample_sigs, sample_labels = next(iter(val_dl))
    sample_sigs = sample_sigs.to("cuda")

    # model = ConvolutionalEcgVAE(DEFAULT_HPARAMS).to("cuda")
    model = ConvolutionalEcgVAE(
        ConvolutionalEcgEncoderDecoderSharedParams(
            seq_len=1000,
            kernel_size=15,
            conv_depth=5,
            latent_dim=20,
            n_filters=16,
        )
    ).to("cuda")
    summary(model, input_data=torch.randn((32, 1, 1000)).to("cuda"))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    initial_beta = 5e-4

    for epoch in range(MAX_EPOCHS):
        print(f"--> Training epoch {epoch}")

        if BETA_ANNEALING:
            # beta annealing, every 100 epochs, increase by an order of magnitude (up to 4)
            beta = min((10 ** (epoch // 100)) * initial_beta, 10)
        else:
            beta = initial_beta

        model.eval()

        # Save sample reconstructions
        save_validation_examples(model, sample_sigs, sample_labels)

        val_combined_loss, val_reproduction_loss, val_kld, val_mean_mse = (
            MeanMetric().to("cuda"),
            MeanMetric().to("cuda"),
            MeanMetric().to("cuda"),
            MeanMetric().to("cuda"),
        )

        for val_batchnum, val_batch in enumerate(val_dl):
            val_sig, val_labels = val_batch
            val_sig = val_sig.to("cuda")

            with torch.no_grad():
                reconstruction, mu, logvar = model(val_sig.unsqueeze(1))
                combined_loss, reproduction_loss, KLD, mean_mse = calculate_losses(
                    reconstruction, val_sig.unsqueeze(1), mu, logvar, beta
                )

            val_combined_loss.update(combined_loss)
            val_reproduction_loss.update(reproduction_loss)
            val_kld.update(KLD)
            val_mean_mse.update(mean_mse)

        curr_val_loss = val_combined_loss.compute()
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss

            print(f"Epoch {epoch:04d} {scheduler.get_last_lr()}(*)")
            model.save("cache/savedmodels/ecgvae.pt")
            # torch.save(model.state_dict(), "cache/savedmodels/ecgvae.pt")

        else:
            print(f"Epoch {epoch:04d} {scheduler.get_last_lr()}")

        print(f"\tBeta: {beta}")
        print(f"\tVal Combined Loss: {val_combined_loss.compute():5f}")
        print(f"\tVal Reproduction Loss: {val_reproduction_loss.compute():5f}")
        print(f"\tVal Mean MSE: {val_mean_mse.compute():5f}")
        print(f"\tVal KLD: {val_kld.compute():5f}")

        model.train()

        for batchnum, batch in enumerate(train_dl):
            sig, labels = batch
            sig = sig.to("cuda")
            # sig = gaussian_smooth1d(sig, sigma=sigma)
            reconstruction, mu, logvar = model(sig.unsqueeze(1))
            loss, _, _, _ = calculate_losses(
                reconstruction, sig.unsqueeze(1), mu, logvar, beta
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

        # scheduler.step(curr_val_loss)
