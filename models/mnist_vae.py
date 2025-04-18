import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels, image_size, latent_dims, channels, kld_weight=1.0):
        super(VAE, self).__init__()
        self.latent_dims = latent_dims
        self.kld_weight = kld_weight
        self.image_size = image_size
        self.channels = channels

        encoder_layers = []
        in_channels = input_channels
        for ch in channels:
            encoder_layers.append(nn.Conv2d(in_channels, out_channels=ch, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            in_channels = ch
        self.encoder = nn.Sequential(*encoder_layers)

        conv_out_size = image_size // (2 ** len(channels))
        self.flatten_dim = channels[-1] * conv_out_size * conv_out_size

        self.mu = nn.Linear(self.flatten_dim, latent_dims)
        self.logvar = nn.Linear(self.flatten_dim, latent_dims)

        self.decoder_input = nn.Linear(latent_dims, self.flatten_dim)

        decoder_layers = []
        reversed_channels = list(reversed(channels))
        in_channels = reversed_channels[0]
        for ch in reversed_channels[1:]:
            decoder_layers.append(nn.ConvTranspose2d(in_channels, ch, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())
            in_channels = ch

        decoder_layers.append(nn.ConvTranspose2d(in_channels, input_channels, kernel_size=4, stride=2, padding=0, output_padding=2))
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), self.channels[-1], self.image_size // (2 ** len(self.channels)), self.image_size // (2 ** len(self.channels)))
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, x, mu, logvar

    def loss_function(self, args):
        x_recon = args[0]
        x = args[1]
        mu = args[2]
        logvar = args[3]

        batch_size = x.size(0)

        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total_loss = recon_loss + self.kld_weight * kld
        
        return total_loss, recon_loss, kld
