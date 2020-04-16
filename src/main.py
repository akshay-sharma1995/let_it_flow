import os, sys
import pdb
import argparse
import torch
import losses
from generator import *
from discriminator import *
import pdb
from utils import *
from torch.utils.data import Dataloader
from torchvision.transforms import transforms
from datetime import datetime
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

    dataloader = DataLoader(dataset, batch_size = 20, shuffle = True, num_workers = 4)


    ## create required directories
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
    
    # train the GAN model
    for epoch in range(num_epochs):
        losses_D = []
        losses_G = []

        data        = np.load('./data/0.npz')
        frames      = data['arr_0']
    
        for batch_ndx, sample in enumerate(dataloader):
            print(sample['frame1'].shape)

            # train discriminator
            
            with torch.no_grad():
                optical_flow = model_gen(frames)
                # frame2_fake = image_warp(X_train[:,0:3],optical_flow,device)

            outDis_real = model_disc(X_train[:,3:])

            
            lossD_real = torch.log(outDis_real)

            outDis_fake = model_disc(frame2_fake)

            lossD_fake = torch.log(1.0 - outDis_fake)
            loss_dis = lossD_real + lossD_fake
            loss_dis = -0.5*loss_dis.mean()

            # calculate customized GAN loss for discriminator
            
            model_disc.optimizer.zero_grad()
            loss_dis.backward()
            optim_dis.step()
            
            losses_D.append(loss_dis.item())

            # train generator
            model_disc.optimizer.zero_grad()

            outDis_fake = model_disc(frame2_fake)
            
            loss_gen = -torch.log(outDis_fake)
            loss_gen = loss_gen.mean()

            model_gen.optimizer.zero_grad() 
            loss_gen.backward()
            model_gen.optimizer.step()

            losses_G.append(loss_gen.item())
            
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

    


if __name__ == "__main__":
    main()

