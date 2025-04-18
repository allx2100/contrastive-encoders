import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_channels, image_size, latent_dims, channels, projections, kld_weight=1.0, contrastive_weight=1.0):
        super(CVAE, self).__init__()
        self.latent_dims = latent_dims
        self.kld_weight = kld_weight
        self.contrastive_weight = contrastive_weight
        self.image_size = image_size
        self.channels = channels

        encoder_layers = []
        in_channels = input_channels
        for ch in channels:
            encoder_layers.append(nn.Conv2d(in_channels, out_channels=ch, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.BatchNorm2d(ch))
            encoder_layers.append(nn.ReLU())
            in_channels = ch
        self.encoder = nn.Sequential(*encoder_layers)

        projection_layers = []
        in_nodes = latent_dims
        for dim in projections:
            projection_layers.append(nn.Linear(in_nodes, dim))
            projection_layers.append(nn.ReLU())
            in_nodes = dim

        self.projection = nn.Sequential(*projection_layers)

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
            decoder_layers.append(nn.BatchNorm2d(ch))
            decoder_layers.append(nn.ReLU())
            in_channels = ch

        decoder_layers.append(nn.ConvTranspose2d(in_channels, input_channels, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid())

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

    def forward(self, x, x_aug=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_proj = self.projection(z)
        x_recon = self.decode(z)

        if x_aug is not None:
            mu_aug, logvar_aug = self.encode(x_aug)
            z_aug = self.reparameterize(mu_aug, logvar_aug)
            z_aug_proj = self.projection(z_aug)
            return x_recon, x, mu, logvar, z, z_proj, z_aug, z_aug_proj
        else:
            return x_recon, x, mu, logvar, z, z_proj, None, None
        
    def contrastive_loss(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        sim_matrix = torch.matmul(z, z.T) 

        sim_matrix /= temperature

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))

        labels = torch.cat([torch.arange(batch_size, device=z.device) + batch_size,
                            torch.arange(batch_size, device=z.device)])

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


    def loss_function(self, args, contrastive=False):
        x_recon, x, mu, logvar, z, z_proj, _, z_aug_proj = args

        batch_size = x.size(0)

        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total_loss = recon_loss + self.kld_weight * kld

        contrastive_loss = torch.tensor(0.0, device=x.device)
        if contrastive and z_aug_proj is not None:
            contrastive_loss = self.contrastive_loss(z_proj, z_aug_proj)
            total_loss += self.contrastive_weight * contrastive_loss

        return total_loss, recon_loss, kld, contrastive_loss

