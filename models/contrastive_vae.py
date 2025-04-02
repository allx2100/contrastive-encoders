import torch
import torch.nn


class CVAE(nn.module):
    def __init__(self, input_channels, image_size, latent_dims, channels, kld_weight):
        super(CVAE, self).__init__()
        self.input_layer = image_size
        self.encoder_net = nn.Sequential(
            nn.Convolution()
        )