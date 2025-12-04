import torch
from torch.utils.data import DataLoader, random_split
from scwfae.data.ptbxlDS import PtbxlDS
from torchinfo import summary
from typing import Tuple
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scwfae.model.ConvolutionalEcgModules import *
from torchmetrics.aggregation import MeanMetric
from torchmetrics import MetricCollection

MAX_EPOCHS = 100
VAL_CHECK_INTERVAL = 500
LEARNING_RATE = 0.0001


def calculate_losses(m: ConvolutionalEcgVAE, batch: Tuple[torch.Tensor, dict]):
    signal, labels = batch
    signal = signal.unsqueeze(1).to("cuda")
    reconstruction, mean, logvar = m(signal)
    # NOTE: the loss reduction for variational autoencoder must be sum
    reproduction_loss = F.mse_loss(reconstruction, signal, reduction="sum")
    mean_mse = F.mse_loss(reconstruction, signal, reduction="mean")
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return reproduction_loss + KLD, reproduction_loss, KLD, mean_mse


def calculate_losses_for_backprop(
    m: ConvolutionalEcgVAE, batch: Tuple[torch.Tensor, dict]
):
    loss, _, _, _ = calculate_losses(m, batch)
    return loss


if __name__ == "__main__":
    torch.manual_seed(42)
    ds = PtbxlDS(lowres=True)

    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])

    train_dl = DataLoader(train_ds, batch_size=32)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds))

    model = ConvolutionalEcgVAE(n_filters=32, latent_dim=2).to("cuda")
    summary(model, input_data=torch.randn((32, 1, 1000)).to("cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    for epoch in range(MAX_EPOCHS):
        print(f"--> Training epoch {epoch}")

        model.eval()

        val_combined_loss, val_reproduction_loss, val_kld, val_mean_mse = (
            MeanMetric().to("cuda"),
            MeanMetric().to("cuda"),
            MeanMetric().to("cuda"),
            MeanMetric().to("cuda"),
        )

        for val_batchnum, val_batch in enumerate(val_dl):
            # Save a sample from the first batch
            if val_batchnum == 0:
                signal, labels = val_batch
                reconstruction, _, _ = model(signal.unsqueeze(1).to("cuda"))
                plt.plot(
                    range(1000),
                    signal[0, :].detach().cpu().numpy(),
                    label="original",
                )
                plt.plot(
                    range(1000),
                    reconstruction[0, :].squeeze().detach().cpu().numpy(),
                    label="reconstruction",
                )
                plt.savefig("./cache/sample.png")
                plt.clf()

            combined_loss, reproduction_loss, KLD, mean_mse = calculate_losses(
                model, val_batch
            )

            val_combined_loss.update(combined_loss)
            val_reproduction_loss.update(reproduction_loss)
            val_kld.update(KLD)
            val_mean_mse.update(mean_mse)

        curr_val_loss = val_combined_loss.compute()
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss

            print(f"Epoch {epoch:04d} {scheduler.get_last_lr()}(*)")
            torch.save(model.state_dict(), "cache/savedmodels/ecgvae.pt")

        else:
            print(f"Epoch {epoch:04d} {scheduler.get_last_lr()}")

        print(f"\tVal Combined Loss: {val_combined_loss.compute():5f}")
        print(f"\tVal Reproduction Loss: {val_reproduction_loss.compute():5f}")
        print(f"\tVal Mean MSE: {val_mean_mse.compute():5f}")
        print(f"\tVal KLD: {val_kld.compute():5f}")

        model.train()

        for batchnum, batch in enumerate(train_dl):
            loss = calculate_losses_for_backprop(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

        scheduler.step(curr_val_loss)
