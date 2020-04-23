import argparse
import pdb
import os,sys
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import skimage
import numpy as np
import visualization_stuff.OpticalFlow_Visualization.flow_vis.flow_vis as flow_vis
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lrd', dest='lr_disc', type=float, default=1e-3, help='learning rate_for_discriminator')
    parser.add_argument('--lrg', dest='lr_gen', type=float, default=1e-3, help='learning rate_for_generator')
    parser.add_argument('--wt-KL', dest='wt_KL', type=float, default=1, help='Weight for KL loss')
    parser.add_argument('--wt-recon', dest='wt_recon', type=float, default=1, help='Weight for Recon loss')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default="../data_scene_flow_multiview/training/image_2/", help='path to data directory')
    parser.add_argument('--data-dir-test', dest='data_dir_test', type=str, default="../data_scene_flow_multiview/testing/image_2/", help='path to test data directory')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--checkpoint', dest='checkpoint_path', type=str, default=None, help='path of a saved_checkpoint')
    parser.add_argument('--train', dest='train', type=int, default=0, help='0 to test the model, 1 to train the model')
    parser.add_argument('--save-interval', dest='save_interval', type=int, default=10, help='epochs after which save the model')
    parser.add_argument('--loaded-epoch', dest='loaded_epoch', type=int, default=1, help='Loading epoch of trained model')
    return parser.parse_args()


def save_model(model, epoch, optimizer, path):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return epoch


def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)

def image_warp(im,flow):
    """Performs a backward warp of an image using the predicted flow.
    Args:
            im: Batch of images. [num_batch, channels, height, width]
            flow: Batch of flow vectors. [num_batch, 2, height, width]
    Returns:
            warped: transformed image of the same shape as the input image.
    """

    ## may think of swaping image axes
    # cuda0 = torch.device('cuda:0')
    im = im.permute(0,2,3,1) ## [num_batch, height, width, channels]
    device = im.device
    flow = flow.permute(0,2,3,1)
    num_batch, height, width, channels = im.size()
    max_x = int(width - 1)
    max_y = int(height - 1)
    zero = torch.zeros([],dtype=torch.int32).to(device).long()

    im_flat = torch.reshape(im,(-1,channels))
    flow_flat = torch.reshape(flow,(-1,2))

    flow_floor = torch.floor(flow_flat).long()
    bilinear_weights = flow_flat - torch.floor(flow_flat)

    pos_x = torch.arange(width).repeat(height*num_batch).to(device).long()
    grid_y = torch.arange(height).unsqueeze(1).repeat(1,width).to(device).long()
    pos_y = torch.reshape(grid_y,(-1,)).repeat(num_batch).long()

    x = flow_floor[:,0]
    y = flow_floor[:,1]
    xw = bilinear_weights[:,0]
    yw = bilinear_weights[:,1]

    wa = ((1-xw)*(1-yw)).unsqueeze(1)
    wb = ((1-xw)*yw).unsqueeze(1)
    wc = (xw*(1-yw)).unsqueeze(1)
    wd = (xw*yw).unsqueeze(1)

    x0 = pos_x + x
    x1 = x0 + 1
    y0 = pos_y + y
    y1 = y0 + 1

    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    dim1 = width * height
    batch_offsets = torch.arange(num_batch).to(device).long() * dim1
    base_grid = batch_offsets.unsqueeze(1).repeat(1,dim1)
    base = torch.reshape(base_grid, (-1,)).long()

    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_a = idx_a.unsqueeze(1).repeat(1,channels)
    idx_b = base_y1 + x0
    idx_b = idx_b.unsqueeze(1).repeat(1,channels)
    idx_c = base_y0 + x1
    idx_c = idx_c.unsqueeze(1).repeat(1,channels)
    idx_d = base_y1 + x1
    idx_d = idx_d.unsqueeze(1).repeat(1,channels)

    # print(im_flat.size())
    # print(idx_a.size())

    Ia = torch.gather(im_flat, dim=0, index=idx_a.long())
    Ib = torch.gather(im_flat, dim=0, index=idx_b.long())
    Ic = torch.gather(im_flat, dim=0, index=idx_c.long())
    Id = torch.gather(im_flat, dim=0, index=idx_d.long())

    warped_flat = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)
    warped = torch.reshape(warped_flat,(num_batch,height,width,channels))
    warped = warped.permute(0,3,1,2)
    im = im.permute(0,3,1,2)
    flow = flow.permute(0,3,1,2)

    return warped

def warp( x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2) (Pytorch tensor)
    flo: [B, 2, H, W] flow (Pytorch tensor)
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
            grid = grid.to(x.device)
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.shape).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:

            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask

def save_samples(frames,dir_name, epoch, folder_name ):
    # pred_frames are numpy ndarray of size (B, 1, H, W)
    save_dir = os.path.join(dir_name, "{}_{}".format(folder_name,epoch))
    make_dirs([save_dir])
    frames = (frames+1)*(255.0/2)
    frames = frames.astype('uint8')
    for i in range(frames.shape[0]):
        fname = os.path.join(save_dir, "{}.png".format(i))
        skimage.io.imsave(fname,frames[i,0])


def save_flow(flow, dir_name, epoch, folder_name):
    save_dir = os.path.join(dir_name, "{}_{}".format(folder_name,epoch))
    make_dirs([save_dir])
    max_u, min_u = flow[:,0].max(), flow[:,0].min()
    max_v, min_v = flow[:,1].max(), flow[:,1].min()
    
    # flow[:,0] = 2*(flow[:,0]-min_u) / (max_u - min_u) - 1.0
    # flow[:,1] = 2*(flow[:,1]-min_v) / (max_v - min_v) - 1.0

    flow[:,0] = (flow[:,0]+1) / 2*(flow.shape[2]-1)
    flow[:,1] = (flow[:,1]+1) / 2*(flow.shape[3]-1)
    # flow = np.moveaxis(flow, [0,1,2,3], [0,2,3,1])
    flow = np.swapaxes(flow, 1,2)
    flow = np.swapaxes(flow, 2,3)
    for i in range(flow.shape[0]):
        flow_color = flow_vis.flow_to_color(flow[i], convert_to_bgr=False)
        fname = os.path.join(save_dir, "{}.png".format(i))
        skimage.io.imsave(fname,flow_color)

    
def scale_grads(parameters, scale):
    for param in parameters:
        param.grad *= scale

# def conv(c_in, c_out, K, S, P=None, d=None , activations=nn.ReLU(), batchnorm=True):
    
        # layers_list = 
        # layer = nn.Sequential(


def plot_prop(data, prop_name, save_path):
    fig = plt.figure(figsize=(16,9))
    plt.plot(data)
    plt.xlabel("epochs")
    plt.ylabel(prop_name)
    plt.savefig(os.path.join(save_path,prop_name+".png"))
    
    plt.close()

def plot_props(data_arr, prop_names, save_path):
    for data, prop_name in zip(data_arr,prop_names):
        plot_prop(data, prop_name, save_path)

def normalize_mag(mag, alpha=0, beta=255):
    mag_max = np.amax(mag, axis=(1,2,3), keepdims=True)
    mag_min = np.amin(mag, axis=(1,2,3), keepdims=True)

    mag = (beta-alpha)*(mag-mag_min)/(mag_max-mag_min) + alpha

    return mag

def save_flow_cv2(flow, dir_name, epoch, folder_name):
    save_dir = os.path.join(dir_name, "{}_{}".format(folder_name,epoch))
    make_dirs([save_dir]) 
    hsv = np.zeros((flow.shape[0],3, flow.shape[2], flow.shape[3]), dtype=np.uint8)
    hsv[:,1,:,:] = 255

    mag = np.sum(flow**2, 1, keepdims=True)**0.5
    ang = np.zeros_like(mag)
    ang = np.arctan2(flow[:,1:2,:,:], flow[:,0:1,:,:])
    hsv[:,0:1,:,:] = ang*180 / (2*np.pi)
    hsv[:,2:3,:,:] = normalize_mag(mag, 0, 255)

    hsv = np.swapaxes(hsv, 1,2)
    hsv = np.swapaxes(hsv, 2,3)

    for i in range(flow.shape[0]):
        bgr = cv2.cvtColor(hsv[i], cv2.COLOR_HSV2BGR)
        fname = os.path.join(save_dir, "{}.png".format(i))
        skimage.io.imsave(fname,bgr)

##################################################################
# unflow utis

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1,inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.1,inplace=True)
                )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1,inplace=True)
            )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                        input2,
                                        kernel_size=1,
                                        patch_size=21,
                                        stride=1,
                                        padding=0,
                                        dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
############################################
