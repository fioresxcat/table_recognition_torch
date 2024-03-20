import argparse
from easydict import EasyDict
import omegaconf
from . import *

if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/slanet.yaml")
    args = parser.parse_args()

    # load config
    config = EasyDict(omegaconf.OmegaConf.load(args.config))

    # train
    trainer = VastTrainer(config)
    trainer.train()