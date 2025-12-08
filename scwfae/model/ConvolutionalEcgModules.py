import torch
from torch.nn import (
    Conv1d,
    LeakyReLU,
    BatchNorm1d,
    MaxPool1d,
    Linear,
    ConvTranspose1d,
    Upsample,
    Sequential,
    Flatten,
    Unflatten,
    Dropout,
)

from typing import Optional

from torchinfo import summary
from dataclasses import dataclass, asdict
from typing import Optional, Self


# For debugging sequentials
def print_sizes(model, input_tensor):
    output = input_tensor
    for m in model.children():
        output = m(output)
        print(m, output.shape)
    return output


@dataclass
class ConvolutionalEcgEncoderDecoderSharedParams:
    """
    Util class that allows various architecture params to be computed only once
    and shared between encoder and decoder

    Ensures symmetry of architecture between encoder and decoder
    """

    seq_len: int
    kernel_size: int
    conv_depth: int
    latent_dim: int
    n_filters: int

    def __post_init__(self):
        self.input_padding_required = False

        if not (self.seq_len % (2**self.conv_depth) == 0):
            self.input_padding_required = True
            pad_amount = (2**self.conv_depth) - (self.seq_len % (2**self.conv_depth))

            print(
                f"Warning: seq_len {self.seq_len} not divisible by 2 ** {self.conv_depth}, will pad up to {self.seq_len + pad_amount}"
            )

            self.seq_len_padded = self.seq_len + pad_amount
            self.left_pad = pad_amount // 2
            self.right_pad = pad_amount - self.left_pad
        else:
            self.seq_len_padded = self.seq_len

        self.layer_padding = (self.kernel_size - 1) // 2

        self.conv_output_size = (
            self.seq_len_padded // (2 ** (self.conv_depth + 1))
        ) * self.n_filters

    def get_conv_padding(self) -> int:
        return self.layer_padding


class ConvolutionalEcgEncoder(torch.nn.Module):

    def __init__(
        self,
        shared_params: ConvolutionalEcgEncoderDecoderSharedParams,
        batchnorm: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.architecture_params = shared_params
        self.batchnorm = batchnorm
        self.dropout = dropout  # TODO

        layers = list()

        # First layer
        layers += [
            Conv1d(
                in_channels=1,
                out_channels=self.architecture_params.n_filters,
                kernel_size=self.architecture_params.kernel_size,
                stride=2,
                padding=self.architecture_params.get_conv_padding(),
            ),
            LeakyReLU(),
        ]

        # Build Additional Conv layers
        for idx in range(0, self.architecture_params.conv_depth):

            layers += [
                Conv1d(
                    in_channels=self.architecture_params.n_filters,
                    out_channels=self.architecture_params.n_filters,
                    kernel_size=self.architecture_params.kernel_size,
                    stride=2,
                    padding=self.architecture_params.get_conv_padding(),
                ),
                LeakyReLU(),
            ]

            if self.batchnorm:
                layers.append(
                    BatchNorm1d(num_features=(2 * self.architecture_params.n_filters))
                )

        layers.append(Flatten(start_dim=1, end_dim=-1))

        layers += [
            Linear(
                self.architecture_params.conv_output_size,
                self.architecture_params.latent_dim,
            ),
            LeakyReLU(),
        ]

        self.net = Sequential(*layers)

    def forward(self, x):
        if self.architecture_params.input_padding_required:
            x = torch.nn.functional.pad(
                x,
                pad=(
                    self.architecture_params.left_pad,
                    self.architecture_params.right_pad,
                ),
                value=0.0,
            )

        return self.net(x)


class ConvolutionalEcgDecoder(torch.nn.Module):

    def __init__(
        self,
        shared_params: ConvolutionalEcgEncoderDecoderSharedParams,
        batchnorm: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.batchnorm = batchnorm
        self.dropout = dropout
        self.padding_required = False

        self.architecture_params = shared_params

        layers = list()
        layers += [
            Linear(
                self.architecture_params.latent_dim,
                self.architecture_params.conv_output_size,
            ),
            LeakyReLU(),
        ]

        layers.append(
            Unflatten(
                dim=1,
                unflattened_size=(
                    self.architecture_params.n_filters,
                    -1,
                ),
            ),
        )

        for idx in range(self.architecture_params.conv_depth, 0, -1):

            layers += [
                ConvTranspose1d(
                    in_channels=self.architecture_params.n_filters,
                    out_channels=self.architecture_params.n_filters,
                    kernel_size=self.architecture_params.kernel_size,
                    padding=self.architecture_params.get_conv_padding(),
                    stride=2,
                    output_padding=1,
                ),
                LeakyReLU(),
            ]

            if self.batchnorm:
                layers.append(
                    BatchNorm1d(num_features=self.architecture_params.n_filters)
                )

        layers += [
            ConvTranspose1d(
                in_channels=self.architecture_params.n_filters,
                out_channels=1,
                kernel_size=self.architecture_params.kernel_size,
                padding=self.architecture_params.get_conv_padding(),
                stride=2,
                output_padding=1,
            )
        ]

        self.net = Sequential(*layers)

    def forward(self, x):
        out = self.net(x)

        if self.architecture_params.input_padding_required:
            return out[
                :,
                :,
                self.architecture_params.left_pad : -self.architecture_params.right_pad,
            ].contiguous()
        else:
            return out


class ConvolutionalEcgVAE(torch.nn.Module):

    def __init__(self, params: ConvolutionalEcgEncoderDecoderSharedParams):
        super(ConvolutionalEcgVAE, self).__init__()
        self.architecture_params = params

        self.encoder = ConvolutionalEcgEncoder(self.architecture_params)
        self.decoder = ConvolutionalEcgDecoder(self.architecture_params)

        self.fc_mean = torch.nn.Linear(
            self.architecture_params.latent_dim,
            self.architecture_params.latent_dim,
        )
        self.fc_logvar = torch.nn.Linear(
            self.architecture_params.latent_dim,
            self.architecture_params.latent_dim,
        )

    def _get_mean_logvar(self, sig: torch.Tensor):
        encoded = self.encoder(sig)

        mu = self.fc_mean(encoded)
        log_var = self.fc_logvar(encoded)

        return mu, log_var

    def vae_encode(self, sig: torch.Tensor):
        mu, log_var = self._get_mean_logvar(sig)
        return self.reparametrize(mu, log_var)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, sig: torch.Tensor):
        mu, log_var = self._get_mean_logvar(sig)
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var

    def save(self, path: str):
        state_dict = self.state_dict()
        state_dict["config"] = asdict(self.architecture_params)
        torch.save(state_dict, path)

    @classmethod
    def load(cls, path: str) -> Self:
        data = torch.load(path, weights_only=True)
        m = cls(ConvolutionalEcgEncoderDecoderSharedParams(**data.pop("config")))
        m.load_state_dict(data)
        return m


if __name__ == "__main__":
    sp = ConvolutionalEcgEncoderDecoderSharedParams(
        seq_len=1000,
        kernel_size=15,
        conv_depth=5,
        latent_dim=25,
        n_filters=16,
    )

    encoder = ConvolutionalEcgEncoder(
        shared_params=sp,
    ).to("cuda")

    decoder = ConvolutionalEcgDecoder(shared_params=sp).to("cuda")

    summary(encoder, input_size=(8, 1, 1000))

    print()

    summary(decoder, input_size=(8, sp.latent_dim))

    cvae = ConvolutionalEcgVAE(params=sp).to("cuda")

    print()

    summary(cvae, input_size=(8, 1, 1000))

    x = torch.randn((8, 1, 1000)).to("cuda")
    print(f"X:\t\t{x.shape}")
    e = encoder(x)
    print(f"E:\t\t{e.shape}")
    d = decoder(e)
    print(f"D:\t\t{d.shape}")
