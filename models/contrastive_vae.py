import torch
import torch.nn as nn


class CVAE(nn.module):
    def __init__(self, input_channels, image_size, latent_dims, channels, kld_weight,encoderlayers):
        super(CVAE, self).__init__()
        self.input_layer = image_size
        self.encoder_net = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 4, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[0],channels[1],4,stride=2,padding=1),
            nn.ReLU()
        )
        self.encoder = nn.Sequential(encoderlayers)
    def encode(self,x):
        x = self.encoder(x)
        return x

    def decode():
        pass
    def forward(data):
        encoded = self.encoder_net
        encoded = encoded.flatten()
        




