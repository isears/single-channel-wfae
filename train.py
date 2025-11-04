import torch
from torch.utils.data import DataLoader, random_split
from scwfae.data.ptbxlDS import PtbxlDS
from torchinfo import summary
from typing import Tuple
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scwfae.model.ConvolutionalEcgModules import *

MAX_EPOCHS = 100
VAL_CHECK_INTERVAL = 500
LEARNING_RATE = 0.001


def calculate_losses(m: ConvolutionalEcgVAE, batch: Tuple[torch.Tensor, dict]):
    signal, labels = batch
    signal = signal.unsqueeze(1).to("cuda")
    reconstruction, mean, logvar = m(signal)
    # NOTE: the loss reduction for variational autoencoder must be sum
    reproduction_loss = F.mse_loss(reconstruction, signal, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return reproduction_loss + KLD


if __name__ == "__main__":
    torch.manual_seed(42)
    ds = PtbxlDS(lowres=True)

    train_ds, val_ds = random_split(ds, lengths=[0.9, 0.1])

    train_dl = DataLoader(train_ds, batch_size=512)
    val_dl = DataLoader(val_ds, batch_size=1024)

    # model = ConvolutionalEcgAutoencoder().to("cuda")
    model = ConvolutionalEcgVAE().to("cuda")

    summary(model, input_data=torch.randn((32, 1, 1000)).to("cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25_000)
    best_val_loss = float("inf")

    for epoch in range(MAX_EPOCHS):
        print(f"--> Training epoch {epoch}")

        for batchnum, batch in enumerate(train_dl):
            if batchnum % VAL_CHECK_INTERVAL == 0:
                model.eval()
                val_losses = list()

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

                    val_losses.append(calculate_losses(model, val_batch))

                avg_val_loss = sum(val_losses) / len(val_losses)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

                    # TODO: save model somewhere
                    print(f"Step {batchnum:04d} validation loss: {avg_val_loss:5f} (*)")
                else:
                    print(f"Step {batchnum:04d} validation loss: {avg_val_loss:5f}")

                model.train()

            loss = calculate_losses(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            # scheduler.step()
