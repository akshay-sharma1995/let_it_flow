import os, sys
import pdb
import argparse
import torch
import losses
from generator import *
from discriminator import *
import pdb
from utils import *
from torch.utils.data import dataloader
from torchvision.transforms import transforms
from datetime import datetime
from DataLoader import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_arguments()
    
    lr_disc = args.lr_disc
    lr_gen = args.lr_gen
    num_epochs = args.num_epochs
    data_dir = args.data_dir
    data_dir_test = args.data_dir_test
    save_interval = args.save_interval
    loaded_epoch = args.loaded_epoch



    disc_save_path = os.path.join("./results", "disc_lrd_{}".format(lr_disc))
    gen_save_path = os.path.join("./results", "gen_lrg_{}".format(lr_gen))

    ## create generator and discriminator instances
    model_gen = gen().to(DEVICE)
    model_disc = disc().to(DEVICE)

    ## loading models
    load_model(model_disc, model_disc.optimizer, disc_save_path+"epoch_{}.pth".format(loaded_epoch))
    load_model(model_gen, model_gen.optimizer, gen_save_path+"epoch_{}.pth".format(loaded_epoch))


    test_dataset = KITTIDataset(folder_name='../data_scene_flow_multiview/testing/image_2/',
    transform=transforms.Compose([RandomVerticalFlip(), 
        RandomHorizontalFlip(), 
        RandomCrop([320, 896]),
        Normalize(),
        ToTensor()
    ]
    ))

    testloader = DataLoader(test_dataset, batch_size = 20, shuffle = True, num_workers = 4)

    
    RCLoss = nn.L1Loss()

    losses_RR = [] 
    losses_Rec = []

    for batch_ndx, frames in enumerate(testloader):

        # my data 

        frames = frames.to(DEVICE).float()
        frames1 = frames[:,0:1,:,:]
        frames2 = frames[:,1:2,:,:]

        # Test discriminator
        with torch.no_grad():
            optical_flow, mean, logvar = model_gen(frames)
            frame2_fake = warp(frames1,optical_flow)

        loss_recons = RCLoss(frame2_fake, frames2)

        losses_Rec.append(loss_recons.item())

        print("Batch_num: {},  Recons_Loss: {:.4f}".format(
            batch_ndx, losses_Rec[-1]))
    

        losses_RR.append(losses_Rec[-1])

    plt.plot(losses_RR, "Reconstruction_loss")
    plt.xlabel('Num Batch')
    plt.show()



if __name__ == "__main__":
    main()

