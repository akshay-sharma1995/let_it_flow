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
import matplotlib.pyplot as plt
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

    dataloader = DataLoader(dataset, batch_size = 2, shuffle = True, num_workers = 4)

    # create required directories
    results_dir = os.path.join(os.getcwd(), "results")
    models_dir = os.path.join(os.getcwd(), "saved_models")
    
    timestamp =  datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    
    model_name = os.path.join(models_dir, "{}_lrd_{}_lrg_{}_epochs{}.pth".format(timestamp,lr_disc,lr_gen, num_epochs))
    curr_dir = os.path.join(results_dir, timestamp)
    
    make_dirs([results_dir, models_dir, curr_dir])
    

    ## create generator and discriminator instances
    model_gen = gen().to(DEVICE)
    model_disc = disc().to(DEVICE)

    # criterion = nn.BCELoss()

    losses_GG = []
    losses_DD = []
    
# # Optical Flow Sanity Check
#     flow_sample = torch.ones([1,2,320,896], dtype = torch.float32)*0
#     sample = dataset[0]
#     sample_1 = sample[0]
#     sample_1 = sample_1.view(1,1,320,896)
#     plt.show()

#     output = warp(sample_1.float(), flow_sample)
#     imgplot = plt.imshow(output[0][0])
#     plt.show()

    # train the GAN model
    for epoch in range(num_epochs):
        losses_D = []
        losses_G = []

        

        for batch_ndx, frames in enumerate(dataloader):

            # my data 
            # frames =  np.random.randint(0, high=1, size=(4,2,320,896))
            # frames =  torch.tensor(frames).to(DEVICE, dtype=torch.float)
            frames = frames.to(DEVICE).float()
            frames1 = frames[:,0:1,:,:]
            frames2_real = frames[:,1:2,:,:]
            # train discriminator
            # with torch.no_grad():
            optical_flow = model_gen(frames)
            frames2_fake = warp(frames1,optical_flow)

            outDis_real = model_disc(frames2_real)

            lossD_real = torch.log(outDis_real)

            outDis_fake = model_disc(frames2_fake)

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
            
            outDis_fake = model_disc(frames2_fake)
            
            loss_gen = -torch.log(outDis_fake)
            loss_gen = loss_gen.mean()

            model_gen.optimizer.zero_grad() 
            loss_gen.backward()
            model_gen.optimizer.step()

            losses_G.append(loss_gen.item())
        
            print("Epoch: [{}/{}], Batch_num: {}, Discriminator loss: {}, Generator loss: {}".format(
                epoch, num_epochs, batch_ndx, losses_D[-1], losses_G[-1]))
        losses_GG.append(np.mean(np.array(losses_G)))
        losses_DD.append(np.mean(np.array(losses_D)))
        
        print("Epoch: [{}/{}], Discriminator loss: {}, Generator loss: {}".format(
            epoch, num_epochs, losses_DD[-1], losses_GG[-1]))
        # print loss while training
        # if (epoch+1) % 30 == 0:
            # print("Epoch: [{}/{}], Discriminator loss: {}, Generator loss: {}".format(
                # epoch, num_epochs, losses_DD[-1], losses_GG[-1]))


    plt.title("Generator Loss")
    plt.ylabel('Loss')
    plt.xlabel('Num of Epochs')
    plt.plot(losses_G)

    plt.show()

    


if __name__ == "__main__":
    main()

