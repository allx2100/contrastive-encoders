import torch
import torch.nn as nn
import torch.nn.functional as F

class SWAE(nn.Module):
    def __init__(self, input_channels, image_size, latent_dim, channels, swd_weight=1.0, n_proj=50):
        super(SWAE, self).__init__()
        self.latent_dim = latent_dim
        self.swd_weight = swd_weight
        self.n_proj = n_proj
        self.image_size = image_size
        self.channels = channels

        encoder_layers = []
        in_channels = input_channels
        for ch in channels:
            encoder_layers.append(nn.Conv2d(in_channels, ch, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            in_channels = ch
        self.encoder = nn.Sequential(*encoder_layers)

        conv_out_size = image_size // (2 ** len(channels))
        self.flatten_dim = channels[-1] * conv_out_size * conv_out_size
        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        decoder_layers = []
        reversed_channels = list(reversed(channels))
        in_channels = reversed_channels[0]
        for ch in reversed_channels[1:]:
            decoder_layers.append(nn.ConvTranspose2d(in_channels, ch, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())
            in_channels = ch

        decoder_layers.append(nn.ConvTranspose2d(in_channels, input_channels, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), self.channels[-1], self.image_size // (2 ** len(self.channels)), self.image_size // (2 ** len(self.channels)))
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def sliced_wasserstein_distance(self, z, prior_samples):
        device = z.device
        B, D = z.size()

        proj = torch.randn(self.n_proj, D, device=device)
        proj = F.normalize(proj, dim=1)

        z_proj = z @ proj.T
        p_proj = prior_samples @ proj.T

        z_proj_sorted, _ = torch.sort(z_proj, dim=0)
        p_proj_sorted, _ = torch.sort(p_proj, dim=0)

        swd = torch.mean((z_proj_sorted - p_proj_sorted) ** 2)
        return swd

    def loss_function(self, x_recon, x, z):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # Sample prior from N(0, I)
        prior_samples = torch.randn_like(z)
        swd = self.sliced_wasserstein_distance(z, prior_samples)

        total_loss = recon_loss + self.swd_weight * swd
        return total_loss, recon_loss, swd
