from omegaconf import OmegaConf
from typing import Union, Optional
from model.model import BBDM, LBBDM
import click
import torch
import PIL.Image as I
from torchvision.transforms import v2
import numpy as np



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))




@click.command()
@click.option("--config", "-c", help="Path to the config file")
def main(config):

    configs = OmegaConf.load(config)
    # t = v2.Compose(
    #     [
    #         v2.PILToTensor(),
    #         v2.Resize(size=(128,128), antialias=True),
    #         v2.ToDtype(torch.float32),
    #     ]
    # )
    bbmodel = BBDM(configs)
    # x = I.open("model/resources/test_image.jpg")
    # x = torch.unsqueeze(t(x), dim=0)
    x = torch.randn((2, 3, 128, 128))
    # bbmodel(torch.tensor([1,1], dtype=torch.int32), x, x)
    bbmodel.sample(x)
    # l = LBBDM(configs)
    # print(l(torch.tensor([1,1], dtype=torch.int32), x, x))
    


main()
