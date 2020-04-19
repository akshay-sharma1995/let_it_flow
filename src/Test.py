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


    dataset_test = KITTIDataset(folder_name=data_dir_test,
    transform=transforms.Compose([RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomCrop([320, 896]),
        Normalize(),
        ToTensor()
    ]
    ))

    dataloader = DataLoader(dataset_test, batch_size = 64, shuffle = True, num_workers = 4)

    
    RCLoss = nn.L1Loss()
    # criterion = nn.BCELoss()

    losses_GG = []
    losses_DD = []
    losses_RR = [] 
    # mean_fake_probs_arr = []
    # std_fake_probs_arr = []
    # train the GAN model
    # for epoch in range(num_epochs):
    losses_D = []
    losses_G = []
    losses_Rec = []
    # fake_probs = []

    for batch_ndx, frames in enumerate(dataloader):

        # my data 
        # frames =  np.random.randint(0, high=1, size=(4,2,320,896))
        # frames =  torch.tensor(frames).to(DEVICE, dtype=torch.float)
        frames = frames.to(DEVICE).float()
        frames1 = frames[:,0:1,:,:]
        frames2 = frames[:,1:2,:,:]
        # Test discriminator
        with torch.no_grad():
            optical_flow, mean, logvar = model_gen(frames)
            frame2_fake = warp(frames1,optical_flow)

        with torch.no_grad():
            outDis_real = model_disc(frames1)

        with torch.no_grad():
            outDis_fake = model_disc(frame2_fake)

        lossD_real = torch.log(outDis_real)
        lossD_fake = torch.log(1.0 - outDis_fake)
        loss_dis = lossD_real + lossD_fake
        loss_dis = -0.5*loss_dis.mean()

        
        losses_D.append(loss_dis.item())

        # Test generator
        optical_flow, mean, logvar = model_gen(frames)
        frame2_fake = warp(frames1,optical_flow)
        
        with torch.no_grad():
            outDis_fake = model_disc(frame2_fake)
        

        loss_KLD = - 0.5 * torch.sum(1 + logvar - mean*mean - torch.exp(logvar))
        loss_gen = -torch.log(outDis_fake)
        loss_gen = loss_gen.mean()
        loss_recons = RCLoss(frame2_fake, frames2)
        total_gen_loss = loss_gen + loss_recons + loss_KLD

        losses_G.append(total_gen_loss.item())
        losses_Rec.append(loss_recons.item())

        print("Batch_num: {}, Discriminator loss: {:.4f}, Generator loss: {:.4f}, Recons_Loss: {:.4f}".format(
            batch_ndx, losses_D[-1], losses_G[-1], loss_recons))
    
        losses_GG.append(losses_G[-1])
        losses_DD.append(losses_D[-1])
        losses_RR.append(losses_Rec[-1])
        # mean_fake_probs_arr.append(np.mean(fake_probs))
        # std_fake_probs_arr.append(np.std(fake_probs))

        # print("Discriminator loss: {:.4f}, Generator loss: {:.4f}, recons_loss: {:.4f} fake_prob: {:.4f}".format(
        #      losses_DD[-1], losses_GG[-1], losses_RR[-1], mean_fake_probs_arr[-1]))
        # if batch_ndx == 2:
        #     break
        


    plt.plot(losses_GG, "Generator_loss")
    plt.plot(losses_DD, "Discriminator_loss")
    plt.plot(losses_RR, "Reconstruction_loss")
    plt.xlabel('Num Batch')
    plt.show()



if __name__ == "__main__":
    main()

