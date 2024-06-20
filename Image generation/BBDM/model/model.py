from diffusers import UNet2DModel, VQModel
from typing import Union
from omegaconf import DictConfig
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class BBDM(nn.Module):
    def __init__(self, configs: Union[DictConfig]) -> None:
        super().__init__()
        self.unet = UNet2DModel(**(configs.model.BB.params.UNetParams))
        self.timesteps = configs.model.BB.params.num_timesteps
        self.max_var = 1
        self.m_t = torch.sin(torch.linspace(0.01, 0.99, self.timesteps + 1))
        self.variance = 2.0 * (self.m_t - self.m_t**2) * self.max_var
        self.eta = 1
        self.loss_type = configs.model.BB.params.loss_type

    def forward(self, timestep, x, y):
        # add noise & y to x - (1-mt)*x + mt*y + sqrt(var)*noise
        noise = torch.rand_like(x)
        sigma_t = torch.sqrt(self.variance[timestep])

        noisy_image = (
            (1 - self.m_t[timestep].view(-1, 1, 1, 1)) * x
            + self.m_t[timestep].view(-1, 1, 1, 1) * y
            + sigma_t.view(-1, 1, 1, 1) * noise
        )
        nois_added = (
            self.m_t[timestep].view(-1, 1, 1, 1) * (y - x)
            + sigma_t.view(-1, 1, 1, 1) * noise
        )

        # predict noise and y
        pred_noise = self.unet(
            noisy_image,
            timestep,
        )["sample"]

        # calculate loss
        if self.loss_type == "l1":
            recloss = (nois_added - pred_noise).abs().mean()
        elif self.loss_type == "l2":
            recloss = F.mse_loss(nois_added, pred_noise)
        else:
            raise NotImplementedError()

        return recloss

    def predict_x0_from_objective(self, timestep, x, y, objective_recon):
        """
        predict initial image from current timestep
        image at time t - dest image added - predected noise
        xt - y*m_t - sqrt(var)* pred_noise
        """

        x0_recon = (
            x
            - self.m_t[timestep].view(-1, 1, 1, 1) * y
            - torch.sqrt(self.variance[timestep]).view(-1, 1, 1, 1) * objective_recon
        ) / (1.0 - self.m_t[timestep].view(-1, 1, 1, 1))

        return x0_recon

    @torch.no_grad
    def sample(self, cond):
        img = cond
        for ts in range(self.timesteps):
            t = torch.full((img.shape[0],), ts, device=img.device, dtype=torch.long)
            n_t = torch.full(
                (img.shape[0],),
                ts + 1,
                device=img.device,
                dtype=torch.long,
            )

            # img0 is not used
            img_rec = self.unet(img, ts)["sample"]
            img0 = self.predict_x0_from_objective(t, img, cond, img_rec)

            m_t = self.m_t[t].view(-1, 1, 1, 1)
            m_nt = self.m_t[t + 1].view(-1, 1, 1, 1)
            var_t = self.variance[t].view(-1, 1, 1, 1)
            var_nt = self.variance[n_t].view(-1, 1, 1, 1)
            sigma2_t = (
                (var_t - var_nt * (1.0 - m_t) ** 2 / (1.0 - m_nt) ** 2) * var_nt / var_t
            )
            sigma_t = torch.sqrt(sigma2_t) * self.eta
            noise = torch.randn_like(img)
            x_tminus_mean = (
                (1.0 - m_nt) * img_rec
                + m_nt * cond
                + torch.sqrt((var_nt - sigma2_t) / var_t)
                * (img - (1.0 - m_t) * img_rec - m_t * cond)
            )

            print(var_t)
            img = x_tminus_mean + sigma_t * noise
            # print(img)

        return img


class LBBDM(BBDM):
    def __init__(self, configs: DictConfig) -> None:
        super().__init__(configs)
        self.configs = configs
        self.vqmodel = VQModel(**configs.model.VQGAN.params)

    def forward(self, timestep, x, y):
        x_latent = self.vqmodel.encode(x)["latents"]
        y_latent = self.vqmodel.encode(y)["latents"]

        return super().forward(timestep, x_latent.detach(), y_latent.detach())
