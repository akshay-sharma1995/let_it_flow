import os, sys
import pdb
import argparse
import torch
from losses import *
from generator import *
from discriminator import *
from FlownetC import *
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
    h,w = frames.shape[-2:]
    
    optical_flow = [F.interpolate(oflow, (h,w)) for oflow in optical_flow]
    
    weights = [0.32, 0.08, 0.02, 0.01, 0.005]
    
    loss = 0.0
    for i in range(0, len(optical_flow)):
        oflow = optical_flow[i]
        frames2_fake = image_warp(frames1,oflow)
         
        loss += flow_loss(frames1, frames2, frames2_fake, oflow, weights[i])
    
    model_gen.optimizer.zero_grad() 
    loss.backward()
    model_gen.optimizer.step()
    print("optical_flow: min: {}, max: {}".format(optical_flow[-1].min(), optical_flow[-1].max()))
    return optical_flow[-1], frames2_fake, loss.item()
        
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
    
    # dataset = MCLVDataset(folder_name=data_dir,
    # transform=transforms.Compose([RandomVerticalFlip(),
        # RandomHorizontalFlip(),
        # RandomCrop([320, 896]),
        # Normalize(),
        # ToTensor()
    # ]
    # ),
    # diff_frames=2
    # )

    dataloader = DataLoader(dataset, batch_size = 10, shuffle = True, num_workers = 4)

    # create required directories
    if (res_dir):
        results_dir = os.path.join(res_dir, "results_flownet")
    else:
        results_dir = os.path.join(os.getcwd(), "results_flownet")
    # models_dir = os.path.join(os.getcwd(), "saved_models")
    
    timestamp =  datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    curr_dir = os.path.join(results_dir, timestamp)
    
    gen_save_path = os.path.join(curr_dir, "gen_lrg_{}".format(lr_gen))

    make_dirs([results_dir, curr_dir])
    

    ## create generator and discriminator instances
    model_gen = FlowNetC(lr=lr_gen).to(DEVICE)
    
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

