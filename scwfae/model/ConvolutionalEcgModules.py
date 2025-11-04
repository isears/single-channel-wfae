import torch
from torch.nn.functional import relu, max_pool1d_with_indices, max_unpool1d, avg_pool1d
from torch.nn import Conv1d, ConvTranspose1d, Linear
from torchinfo import summary


class ConvolutionalEcgEncoder(torch.nn.Module):
    """
    Two-layer convolutional ECG encoder
    - Expects 10s single channel ECG data (any channel) at 100Hz
    - Relatively large first-layer convolution kernel (7) to smooth out high frequency noise
    - Relatively small second-layer convolution kernel (3) for recognition of higher level features
    - Max pooling between layers w/kernel size equal to preceeding layer
    - seq_len dimension will go from 1000 -> 47
    - Theoretical maximum recognizable number of unique waves: n_filters^2
    """

    def __init__(self, n_filters: int):
        super(ConvolutionalEcgEncoder, self).__init__()

        self.n_filters = n_filters

        self.conv1 = Conv1d(
            in_channels=1, out_channels=n_filters, kernel_size=7, padding=3
        )
        self.conv2 = Conv1d(
            in_channels=n_filters,
            out_channels=n_filters**2,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = relu(out)
        out, first_layer_indices = max_pool1d_with_indices(
            out, kernel_size=(7,), return_indices=True
        )
        out = self.conv2(out)
        out = relu(out)
        out, second_layer_indices = max_pool1d_with_indices(
            out, kernel_size=(3,), return_indices=True
        )

        out = avg_pool1d(out, kernel_size=(47,))

        return out, first_layer_indices, second_layer_indices


class ConvolutionalEcgDecoder(torch.nn.Module):
    """
    Decoder counterpart to encoder

    - Mirror image architecture
    - Supports raw autoencoder design by accepting encoder output directly
        - E.g. reconstruction = decoder(encoder(x))
    - Need intermediate module for variational autoencoder functionality
    - Must be initialized with same number of filters as encoder
    """

    def __init__(self, n_filters: int):
        super(ConvolutionalEcgDecoder, self).__init__()

        self.n_filters = n_filters

        self.conv1 = ConvTranspose1d(
            in_channels=n_filters**2,
            out_channels=n_filters,
            kernel_size=3,
        )

        self.conv2 = ConvTranspose1d(
            in_channels=n_filters, out_channels=1, kernel_size=7
        )

    def forward(
        self,
        x: torch.Tensor,
        first_layer_indices: torch.Tensor,
        second_layer_indices: torch.Tensor,
    ):
        batch_size, n_filters, _ = x.shape
        out = x.expand((batch_size, n_filters, 47))
        out = max_unpool1d(out, indices=second_layer_indices, kernel_size=(3,))
        out = relu(out)
        out = self.conv1(out)

        # Essentially, a reverse pad. Need to drop 5 elements
        # TODO: this is fragile and will break with changing kernels, need to make more robust
        out = out[:, :, :-1]

        out = max_unpool1d(out, indices=first_layer_indices, kernel_size=(7,))
        out = self.conv2(out)
        # TODO: consider better activation layers
        # out = relu(out)

        return out


class ConvolutionalEcgAutoencoder(torch.nn.Module):
    def __init__(self, n_filters: int = 16):
        super(ConvolutionalEcgAutoencoder, self).__init__()
        self.encoder = ConvolutionalEcgEncoder(n_filters)
        self.decoder = ConvolutionalEcgDecoder(n_filters)

    def forward(self, x):
        z, indices1, indices2 = self.encoder(x)
        reconstruction = self.decoder(z, indices1, indices2)

        return reconstruction


class ConvolutionalEcgVAE(torch.nn.Module):
    def __init__(self, n_filters: int = 16, latent_dim: int = 64):
        super(ConvolutionalEcgVAE, self).__init__()

        self.encoder = ConvolutionalEcgEncoder(n_filters)
        self.decoder = ConvolutionalEcgDecoder(n_filters)

        self.fc_mean = Linear(n_filters**2, latent_dim)
        self.fc_logvar = Linear(n_filters**2, latent_dim)
        self.fc_decode = Linear(latent_dim, n_filters**2)

    def encode(self, x):
        enc_raw, _, _ = self.encoder(x)
        mu = self.fc_mean(enc_raw)
        sigma = self.fc_logvar(enc_raw)

        # Reparameterization
        batch, dim = mu.shape
        epsilon = (
            torch.distributions.normal.Normal(0, 1).sample((batch, dim)).to(mu.device)
        )

        return mu + torch.exp(0.5 * sigma) * epsilon

    def forward(self, x):
        enc_raw, indices1, indices2 = self.encoder(x)
        mu = self.fc_mean(enc_raw.squeeze())
        sigma = self.fc_logvar(enc_raw.squeeze())

        # Reparameterization
        batch, dim = mu.shape
        epsilon = (
            torch.distributions.normal.Normal(0, 1).sample((batch, dim)).to(mu.device)
        )

        z = mu + torch.exp(0.5 * sigma) * epsilon

        reconstruction = self.decoder(
            self.fc_decode(z).unsqueeze(-1), indices1, indices2
        )

        return reconstruction, mu, sigma


if __name__ == "__main__":
    e = ConvolutionalEcgEncoder(n_filters=16)
    d = ConvolutionalEcgDecoder(n_filters=16)

    x_sample = torch.rand((32, 1, 1000))

    summary(e, input_data=x_sample)

    x_hat, i1, i2 = e(x_sample)
    print()
    summary(d, input_data=(x_hat, i1, i2))

    ae = ConvolutionalEcgAutoencoder()
    print()
    summary(ae, input_data=(x_sample))

    vae = ConvolutionalEcgVAE()
    print()
    summary(vae, input_data=(x_sample))
