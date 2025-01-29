import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", default='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/configs/surfden_1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,   #######
    # parser.add_argument('--resume', default='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/save_model/ckpts/test_Bmag_ddim_0207_testB30_ddm.pth.tar', type=str, 
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=50,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=42, type=int, metavar='N',
                        help='Seed for initializing training (default: 42)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

                                               
    # torch.backends.cuda.matmul.allow_tf32 = True                                    
    # torch.backends.cudnn.benchmark = True                                           
    # torch.backends.cudnn.deterministic = False                                      
    # torch.backends.cudnn.allow_tf32 = True       

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    # DATASET = datasets.__dict__[config.data.dataset](config)
    DATASET = datasets.AllData(config)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
