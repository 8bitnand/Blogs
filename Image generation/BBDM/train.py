from omegaconf import OmegaConf
from typing import Union, Optional
from model.model import BBDM
import click 

@click.command()
@click.option("--config", "-c", help="Path to the config file")
def main(config):

    configs = OmegaConf.load(config)
    
    bbmodel = BBDM(configs)

main()
