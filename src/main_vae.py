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



def train_gen(frames,frames1, frames2, RCLoss, wt_recon, wt_KL, model_gen):

    optical_flow, mean, logvar = model_gen(frames)
    print("optical_flow: min: {}, max: {}".format(optical_flow.min(), optical_flow.max()))
    frame2_fake = image_warp(frames1,optical_flow)
    
    # calculate losses
    loss_KLD = - 0.5 * torch.sum(1 + logvar - mean*mean - torch.exp(logvar))

    loss_recons = RCLoss(frame2_fake, frames2)
    
    total_gen_loss = wt_recon*loss_recons + wt_KL*loss_KLD

    # update the model
    model_gen.optimizer.zero_grad() 
    total_gen_loss.backward()
    model_gen.optimizer.step()
    
    return optical_flow, frame2_fake, total_gen_loss.item(), loss_recons.item()

def main():
    args = parse_arguments()
    
    lr_disc = args.lr_disc
    lr_gen = args.lr_gen
    num_epochs = args.num_epochs
    data_dir = args.data_dir
    save_interval = args.save_interval
    wt_recon = args.wt_recon
    wt_KL = args.wt_KL
    res_dir = args.results_dir

    dataset = KITTIDataset(folder_name=data_dir,
    transform=transforms.Compose([RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomCrop([320, 896]),
        Normalize(),
        ToTensor()
    ]
    ))

    dataloader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 4)

    # create required directories
    if(res_dir):
        results_dir = os.path.join(res_dir, "results_flownet")
    else:
        results_dir = os.path.join(os.getcwd(), "results_flownet")
    # models_dir = os.path.join(os.getcwd(), "saved_models")
    
    timestamp =  datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    curr_dir = os.path.join(results_dir, timestamp)
    
    # disc_save_path = os.path.join(curr_dir, "disc_lrd_{}".format(lr_disc))
    gen_save_path = os.path.join(curr_dir, "gen_lrg_{}".format(lr_gen))

    make_dirs([results_dir, curr_dir])
    

    ## create generator and discriminator instances
    model_gen = gen().to(DEVICE)
    # model_disc = disc().to(DEVICE)
    
    RCLoss = nn.L1Loss()
    # criterion = nn.BCELoss()

    losses_GG = []
    losses_DD = []
    losses_RR = [] 
    mean_fake_probs_arr = []
    std_fake_probs_arr = []
    # train the GAN model

    save_sample_flag = False
    for epoch in range(num_epochs):
        losses_D = []
        losses_G = []
        losses_Rec = []
        fake_probs = []
        if(epoch%2==0):
            save_sample_flag = True
        for batch_ndx, frames in enumerate(dataloader):

            # process data
            ##########################################
            frames = frames.to(DEVICE).float()
            frames1 = frames[:,0:1,:,:]
            frames2 = frames[:,1:2,:,:]

            # train generator
            #########################################
            optical_flow, frame2_fake, total_gen_loss, loss_recons = train_gen(frames, 
                                                                                frames1, 
                                                                                frames2,  
                                                                                RCLoss,
                                                                                wt_recon,
                                                                                wt_KL, 
                                                                                model_gen)
            losses_G.append(total_gen_loss*1.0)
            losses_Rec.append(loss_recons*1.0)


            # save images, and flow
            ##########################################
            if(save_sample_flag):
                save_samples(frame2_fake.clone().detach().cpu().numpy(), curr_dir, epoch, "predicted")
                save_samples(frames1.cpu().numpy(), curr_dir, epoch, "actual_frame1")
                save_samples(frames2.cpu().numpy(), curr_dir, epoch, "actual_frame2")
                save_flow(optical_flow.clone().detach().cpu().numpy(), curr_dir, epoch, "flow")
                save_sample_flag = False


            print("Epoch: [{}/{}], Batch_num: {}, Generator loss: {:.4f}, Recons_Loss: {:.4f}".format(
                epoch, num_epochs, batch_ndx, losses_G[-1], loss_recons))
    
        losses_GG.append(np.mean(losses_G))
        losses_RR.append(np.mean(losses_Rec))

        print("Epoch: [{}/{}], Generator loss: {:.4f}, recons_loss: {:.4f}".format(
            epoch+1, num_epochs, losses_GG[-1], losses_RR[-1]))

        # save model
        ##################################################
        if (epoch+1) % save_interval == 0:
            save_model(model_gen, epoch, model_gen.optimizer, gen_save_path+"epoch_{}.pth".format(epoch))


    plot_props([losses_GG, losses_RR],
                ["Generator_loss", "Reconstruction_loss"],
                curr_dir)


if __name__ == "__main__":
    main()

