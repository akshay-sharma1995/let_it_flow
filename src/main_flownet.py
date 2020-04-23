import torch
import os
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
import skimage.io
import skimage.color    
# import models
# import datasets
# from multiscaleloss import multiscaleEPE, realEPE
# import datetime
# from tensorboardX import SummaryWriter
import numpy as np
import random
from matplotlib import pyplot as plt
from FlownetC import *
import torch.optim as optim
import torch
from train import *
from util import rgb_to_y

def imlist(fpath):
    flist = os.listdir(fpath)
    return flist

def plot_loss(epoch_loss_list,num_epochs):
    epochs = np.arange(1,num_epochs+1)
    plt.plot(epochs,epoch_loss_list,'k-')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def main():
    model = FlowNetC()
    # s_loss = 
    if torch.cuda.is_available():
        model.cuda()
        print("Model shifted to GPU")

    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    DIR = "../../../Project_data/training_data/npz_3_set/"
    checkpoints_dir = "./checkpoints/"
    # input_transform = transforms.Compose([
    #     flow_transforms.ArrayToTensor(),
    #     transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
    #     transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    # ])
    # target_transform = transforms.Compose([
    #     flow_transforms.ArrayToTensor(),
    #     transforms.Normalize(mean=[0,0],std=[20,20])
    # ])
    batch_size = 1
    # length =
    flist = imlist(DIR)
    # print("flist",flist)
    number_of_image_sets = len(flist)
    idxs = (np.arange(1,number_of_image_sets,1)) ## id of all image_sets
    random.shuffle(idxs) ## shuffling the idxs
    num_epochs = 10
    batches_processed = 0
    epoch_loss_list = []
    for epoch in range(0,num_epochs):

        epoch_loss = 0
        for i in range(len(idxs)):
            image_batch = []
            if(len(idxs)-i>=batch_size):
                count = batch_size
            else:
                count = len(idxs) - i
            for j in range(count): ## making batches
                path = DIR + flist[idxs[i]]
                image_triplet = np.load(path)['arr_0']
                image_triplet[0] = skimage.color.rgb2ycbcr(image_triplet[0])
                image_triplet[1] = skimage.color.rgb2ycbcr(image_triplet[1])
                image_triplet[2] = skimage.color.rgb2ycbcr(image_triplet[2])

                image_triplet = image_triplet[:,:,:,0:1] ## only y channel of each image

                ## image_triplet.size = 3xHxWx3
                ## mapping the images to (-1,1)
                image_triplet = ((image_triplet.astype(np.float64) - 16.0) / (235.0-16.0))   * 2.0 - 1.0
                # print("max_min", np.nanmax(image_triplet),np.nanmin(image_triplet))
                image_batch.append(image_triplet)

            image_batch = np.array(image_batch)
            image_batch = np.rollaxis(image_batch,4,2)

            # print("listofimages.shape",np.shape(image_batch))
            batches_processed += 1
            batch_loss = train(image_batch,model,optimizer)
            epoch_loss += batch_loss
            if((batches_processed%200)==0):
                print("Epoch = ",epoch+1," Loss_after_",batches_processed,"_batches = ",epoch_loss)
        print("Epoch = ",epoch+1,"/ ",num_epochs,"  Loss = ",epoch_loss)
        epoch_loss_list.append(epoch_loss)

            ## saving model_weights
        # if((epoch + 1)%1):
        checkpoint_file_name = checkpoints_dir + "t_loss_model"+str(epoch+1)+".pth"
        torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss}, checkpoint_file_name)
        print("checkpoint_saved")
            

    plot_loss(epoch_loss_list,num_epochs)
if __name__ == '__main__':
    main()
