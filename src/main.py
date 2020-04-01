import os, sys
import pdb
import argparse
import torch
from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    args = parse_args()
    
    lr_disc = args.lr_disc
    lr_gen = args.lr_gen
    num_epochs = args.num_epochs

    
    ### TODO:
    # make directories
    # make network instances
    # load checkpoints if needed
    # initialize all props to be saved
    # train function
    # test function


if __name__=="__main":
    main(sys.argv)
