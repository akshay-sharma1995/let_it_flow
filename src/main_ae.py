import os, sys
import pdb
import argparse
import torch
from losses import *
from generator import *
from discriminator import *
from autoencoder import *
import pdb
from utils import *
from torch.utils.data import dataloader
from torchvision.transforms import transforms
from datetime import datetime
from DataLoader import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_generator(frames,frames1, frames2, model_gen):
    
    # generator forward pass
    optical_flow = model_gen(frames)
    print("optical_flow: min: {}, max: {}".format(optical_flow.min(), optical_flow.max()))
    # frame2_fake = warp(frames1,optical_flow)
    frames2_fake = image_warp(frames1, optical_flow)

    total_gen_loss = flow_loss(frames1, frames2, frames2_fake, optical_flow)

    # update the model
    model_gen.optimizer.zero_grad() 
    total_gen_loss.backward()
    model_gen.optimizer.step()
    
    return optical_flow, frames2_fake, total_gen_loss.item()
        
def main():
    args = parse_arguments()
    
    lr_disc = args.lr_disc
    lr_gen = args.lr_gen
    num_epochs = args.num_epochs
    data_dir = args.data_dir
    save_interval = args.save_interval
    wt_recon = args.wt_recon
    wt_KL = args.wt_KL


    # dataset = KITTIDataset(folder_name=data_dir,
    # transform=transforms.Compose([RandomVerticalFlip(),
        # RandomHorizontalFlip(),
        # RandomCrop([320, 896]),
        # Normalize(),
        # ToTensor()
    # ]
    # ))
    
    dataset = MCLVDataset(folder_name=data_dir,
    transform=transforms.Compose([RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomCrop([320, 896]),
        Normalize(),
        ToTensor()
    ]
    ),
    diff_frames=2
    )

    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 4)

    # create required directories
    results_dir = os.path.join(os.getcwd(), "results_autoencoder")
    # models_dir = os.path.join(os.getcwd(), "saved_models")
    
    timestamp =  datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    curr_dir = os.path.join(results_dir, timestamp)
    
    gen_save_path = os.path.join(curr_dir, "gen_lrg_{}".format(lr_gen))

    make_dirs([results_dir, curr_dir])
    

    ## create generator and discriminator instances
    model_gen = auto_enc().to(DEVICE)
    
    losses_GG = []
    # train the GAN model

    save_sample_flag = False
    for epoch in range(num_epochs):
        losses_D = []
        losses_G = []
        losses_Rec = []
        fake_probs = []
        if(epoch%10==0):
            save_sample_flag = True
        for batch_ndx, frames in enumerate(dataloader):

            # my data 
            frames = frames.to(DEVICE).float()
            frames1 = frames[:,0:1,:,:]
            frames2 = frames[:,1:2,:,:]
            
            
            ## train generator
            ###############################################################
            optical_flow,  frames2_fake, loss_gen = train_generator(frames,
                                                                    frames1, 
                                                                    frames2, 
                                                                    model_gen)

            losses_G.append(loss_gen)
             
            # save samples
            #############################################################
            if(save_sample_flag):
                save_samples(frames2_fake.clone().detach().cpu().numpy(), curr_dir, epoch, "predicted")
                save_samples(frames1.cpu().numpy(), curr_dir, epoch, "actual_frame1")
                save_samples(frames2.cpu().numpy(), curr_dir, epoch, "actual_frame2")
                save_flow(optical_flow.clone().detach().cpu().numpy(), curr_dir, epoch, "flow")
                save_sample_flag = False
            
            print("Epoch: [{}/{}], Batch_num: {}, loss: {:.4f},".format(epoch, num_epochs, batch_ndx, losses_G[-1]))
            
    
        losses_GG.append(np.mean(losses_G))

        print("Epoch: [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, losses_GG[-1]))
        
        if (epoch+1) % save_interval == 0:
            save_model(model_gen, epoch, model_gen.optimizer, gen_save_path+"epoch_{}.pth".format(epoch))

    plot_props([losses_GG],
                ["flow_loss"],
                curr_dir)


if __name__ == "__main__":
    main()

