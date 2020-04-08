import os, sys
import pdb
import argparse
import torch
import losses
# # from utils import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
# Command-line flags are defined here.
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                                                default=100, help="Number of epochs")

        parser.add_argument("--lr_disc",dest="lr_disc",type=float,
                                                default=5e-4,help="Discriminator Learning Rate")

        parser.add_argument('--lr_gen', dest='lr_gen', type=float,
                                                default=5e-4, help="Generator Learning Rate")
        return parser.parse_args()

def main():
    args = parse_arguments()
    
    lr_disc = args.lr_disc
    lr_gen = args.lr_gen
    num_epochs = args.num_epochs

    print('num_epochs')

#     ### TODO:
#     # make directories
#     # make network instances
#     # load checkpoints if needed
#     # initialize all props to be saved
#     # train function
#     # test function


if __name__ == "__main__":
    main()

