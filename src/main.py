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

    model_gen = gen().to(DEVICE)
    model_disc = disc().to(DEVICE)

    criterion = nn.BCELoss()

    losses_GG = []
    losses_DD = []
    
    # train the GAN model
    for epoch in range(num_epochs):
        losses_D = []
        losses_G = []

        data        = np.load('./data/0.npz')
        frames      = data['arr_0']


        frames = np.swapaxes(frames,0,2)
        frames = np.swapaxes(frames,1,2)
        frames = np.expand_dims(frames,axis=0)
        frames = torch.tensor(frames, device = DEVICE).float()
        frame1      = frames[:,0:1,:,:]
        frame2_real = frames[:,1:2,:,:]
                
        # train discriminator
        model_gen.optimizer.zero_grad()
        outDis_real = model_disc(frame2_real)
        label_real  = torch.ones([frame2_real.shape[0],1]).to(device)
        lossD_real  = criterion(outDis_real, label_real)
        
        optical_flow = model_gen(frames)
        frame2_fake = image_warp(frame1,optical_flow,device)

        outDis_fake = model_disc(frame2_fake)
        label_fake  = torch.zeros([frame2_real.shape[0],1]).to(device)
        lossD_fake  = criterion(outDis_fake, label_fake)

        loss_dis = lossD_real + lossD_fake
        losses_D.append(loss_dis.item())

        # calculate customized GAN loss for discriminator
        
        loss_dis.backward(retain_graph=True)
        optim_dis.step()

        # train generator
        model_disc.optimizer.zero_grad()

        outDis_fake = model_disc(frame2_fake)
        label_real  = torch.ones([y_real.shape[0],1]).to(device)
        loss_gen  = criterion(outDis_fake, label_real)

        losses_G.append(loss_gen.item())
        
        # pdb.set_trace()

        
        loss_gen.backward()
        optim_gen.step()

        # print loss while training
        if (n_batch + 1) % 30 == 0:
            print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {}, Generator loss: {}".format(
                epoch, num_epochs, n_batch, loss_dis.item(), loss_gen.item()))
        losses_GG.append(np.mean(np.array(losses_G)))
        losses_DD.append(np.mean(np.array(losses_D)))

    plt.title("Generator Loss")
    plt.ylabel('Loss')
    plt.xlabel('Num of Epochs')
    plt.plot(losses_G)

    plt.show()

    

#     ### TODO:
#     # make directories
#     # make network instances
#     # load checkpoints if needed
#     # initialize all props to be saved
#     # train function
#     # test function


if __name__ == "__main__":
    main()

