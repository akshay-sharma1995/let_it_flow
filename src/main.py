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
    save_interval = args.save_interval


    dataset = KITTIDataset(folder_name=data_dir,
    transform=transforms.Compose([RandomCrop([320, 896]),
        Normalize(),
        ToTensor()
    ]
    ))

    dataloader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 4)

    # create required directories
    results_dir = os.path.join(os.getcwd(), "results")
    # models_dir = os.path.join(os.getcwd(), "saved_models")
    
    timestamp =  datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    curr_dir = os.path.join(results_dir, timestamp)
    
    disc_save_path = os.path.join(curr_dir, "disc_lrd_{}".format(lr_disc))
    gen_save_path = os.path.join(curr_dir, "gen_lrg_{}".format(lr_gen))

    make_dirs([results_dir, curr_dir])
    

    ## create generator and discriminator instances
    model_gen = gen().to(DEVICE)
    model_disc = disc().to(DEVICE)
    
    RCLoss = nn.L1Loss()
    # criterion = nn.BCELoss()

    losses_GG = []
    losses_DD = []
    losses_RR = [] 
    mean_fake_probs_arr = []
    std_fake_probs_arr = []
    # train the GAN model
    for epoch in range(num_epochs):
        losses_D = []
        losses_G = []
        losses_Rec = []
        fake_probs = []

        for batch_ndx, frames in enumerate(dataloader):

            # my data 
            # frames =  np.random.randint(0, high=1, size=(4,2,320,896))
            # frames =  torch.tensor(frames).to(DEVICE, dtype=torch.float)
            frames = frames.to(DEVICE).float()
            frames1 = frames[:,0:1,:,:]
            frames2 = frames[:,1:2,:,:]
            # train discriminator
            with torch.no_grad():
                optical_flow = model_gen(frames)
                frame2_fake = warp(frames1,optical_flow)

            outDis_real = model_disc(frames1)

            
            lossD_real = torch.log(outDis_real)

            outDis_fake = model_disc(frame2_fake)

            lossD_fake = torch.log(1.0 - outDis_fake)
            loss_dis = lossD_real + lossD_fake
            loss_dis = -0.5*loss_dis.mean()

            # calculate customized GAN loss for discriminator
            
            model_disc.optimizer.zero_grad()
            loss_dis.backward()
            model_disc.optimizer.step()
            
            losses_D.append(loss_dis.item())

            # train generator
            model_disc.optimizer.zero_grad()

            outDis_fake = model_disc(frame2_fake)
            

            loss_gen = -torch.log(outDis_fake)
            loss_gen = loss_gen.mean()

            loss_recons = RCLoss(frame2_fake, frames2)
            
            total_gen_loss = loss_gen + loss_recons

            model_gen.optimizer.zero_grad() 
            total_gen_loss.backward()
            model_gen.optimizer.step()

            losses_G.append(loss_gen.item())
            losses_Rec.append(loss_recons.item())
            fake_probs.extend(outDis_fake.clone().detach().cpu().numpy())
            
            print("Epoch: [{}/{}], Batch_num: {}, Discriminator loss: {:.4f}, Generator loss: {:.4f}, Recons_Loss: {:.4f}".format(
                epoch, num_epochs, batch_ndx, losses_D[-1], losses_G[-1], loss_recons))
    
        losses_GG.append(np.mean(losses_G))
        losses_DD.append(np.mean(losses_D))
        losses_RR.append(np.mean(losses_Rec))
        mean_fake_probs_arr.append(np.mean(fake_probs))
        std_fake_probs_arr.append(np.std(fake_probs))

        print("Epoch: [{}/{}], Discriminator loss: {:.4f}, Generator loss: {:.4f}, recons_loss: {:.4f} fake_prob: {:.4f}".format(
            epoch+1, num_epochs, losses_DD[-1], losses_GG[-1], losses_RR[-1], mean_fake_probs_arr[-1]))
        
        if (epoch+1) % 2 == 0:
            save_model(model_disc, epoch, model_disc.optimizer, disc_save_path+"epoch_{}.pth".format(epoch))
            save_model(model_gen, epoch, model_gen.optimizer, gen_save_path+"epoch_{}.pth".format(epoch))

    plot_props([losses_GG, losses_DD, losses_RR, mean_fake_probs_arr],
                ["Generator_loss", "Discriminator_loss", "Reconstruction_loss", "disc_fake_prob"],
                curr_dir)


if __name__ == "__main__":
    main()

