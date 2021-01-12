from typing import List, TypeVar

import torch
from torch import nn

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

from torch.nn.utils import spectral_norm

class LinearUpsample(nn.Module):

    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.upsampler = nn.Linear(self.layer_size * self.layer_size,
                                   self.layer_size * self.layer_size * 4)
    def forward(self, input):
        _, ch, _, _ = input.shape
        result = input.view(-1, ch, self.layer_size * self.layer_size)
        result = self.upsampler(result)
        result = result.view(-1, ch, self.layer_size * 2, self.layer_size * 2)
        return result


class SimpleVaeLinear2Deep(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel : int,
                 latent_dim: int = 128,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(SimpleVaeLinear2Deep, self).__init__()

        self.last_layer_size = 64

        # self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channel = h_dim

        self.encoder = nn.Sequential(*modules)
        # self.fc = nn.Linear(hidden_dims[-1] * self.last_layer_size*self.last_layer_size, latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1] * self.last_layer_size*self.last_layer_size, latent_dim)

        # Build Decoder
        modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.last_layer_size*self.last_layer_size)
        self.upsampler = nn.Linear(self.last_layer_size*self.last_layer_size, self.last_layer_size*self.last_layer_size*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    LinearUpsample(hidden_dims[i]),
                    nn.Conv2d(hidden_dims[i], out_channels=hidden_dims[i],
                              kernel_size=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            # nn.Conv2d(hidden_dims[-1], out_channels=hidden_dims[-1],
            #           kernel_size=1),
            # nn.BatchNorm2d(h_dim),
            # nn.ReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channel,
                      kernel_size=1)
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print(f"input:encode:{input.shape}")
        result = self.encoder(input)
        # print(f"result:{result.shape}")
        # result = torch.flatten(result, start_dim=1)
        # print(f"flatten result:{result.shape}")

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # mu = self.fc(result)
        # log_var = self.fc_var(result)

        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # print(f"input,decode_input:{z.shape}")
        # result = self.decoder_input(z)
        # print(f"result:{result.shape}")
        # result = z.view(-1, 32, self.last_layer_size, self.last_layer_size)
        # print(f"result:{result.shape}")
        result = self.decoder(z)
        # print(f"decode:{result.shape}")

        result = self.final_layer(result)
        # print(f"final:{result.shape}")
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
    #     mu, log_var = self.encode(input)
    #     z = self.reparameterize(mu, log_var)
    #     return  [self.decode(z), input, mu, log_var]
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu = self.encode(input)
        # z = self.reparameterize(mu, log_var)
        return self.decode(mu)
