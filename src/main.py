import os, sys
import pdb
import argparse
import torch
import losses
from generator import *
from discriminator import *
import pdb
from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_arguments()
    
    lr_disc = args.lr_disc
    lr_gen = args.lr_gen
    num_epochs = args.num_epochs

    data  = np.load('./data/0.npz')
    frames = data['arr_0']
    frame1 = frames[:,:,0]
    frame2 = frames[:,:,1]

    pdb.set_trace()
    frames = np.swapaxes(frames,0,2)
    frames = np.swapaxes(frames,1,2)
    frames = np.expand_dims(frames,axis=0)


    model_gen = gen()
    model_disc = disc()

    frames = torch.tensor(frames, device = DEVICE).float()
    optical_flow = model_gen(frames).to(device)

    frame2_fake = image_warp(frame1,optical_flow,device)

    

#     ### TODO:
#     # make directories
#     # make network instances
#     # load checkpoints if needed
#     # initialize all props to be saved
#     # train function
#     # test function


if __name__ == "__main__":
    main()

