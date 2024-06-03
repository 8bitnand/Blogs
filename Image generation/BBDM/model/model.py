from diffusers import UNet2DModel, VQModel
import torch
from typing import Union
from omegaconf import OmegaConf
import numpy as np


class BBDM(torch.nn.Module):
    def __init__(self, configs: Union[dict]) -> None:
        super().__init__()
        self.unet = UNet2DModel(**vars(configs.model.BB.params.UNetParams))
        self.timesteps = configs.model.BB.params.num_timesteps
        self.max_var = 1
        self.m_t = torch.sin(torch.linspace(0.01, 0.99, self.timesteps))
        self.m_tminus = torch.cat([torch.tensor([0.0]), self.m_t[:-1]])
        self.variance = 2.0 * (self.m_t - self.m_t**2) * self.max_var

    def forward(self, timestep, x, y):
        # add noise & y to x - (1-mt)*x + mt*y + sqrt(var)*noise
        noise = torch.rand_like(x)
        noisy_image = (
            (1 - self.m_t[timestep]) * x
            + self.m_t * y
            + torch.sqrt(self.variance[timestep]) * noise
        )

        # predict noise and y

        # calculate loss
