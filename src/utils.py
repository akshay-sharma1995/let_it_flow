import argparse
import pdb
import os,sys
import torch
import torch.nn as nn
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lrd', dest='lr_disc', type=float, default=1e-3, help='learning rate_for_discriminator')
    parser.add_argument('--lrg', dest='lr_gen', type=float, default=1e-3, help='learning rate_for_generator')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default="../dataset/train_data/", help='path to data directory')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--checkpoint', dest='checkpoint_path', type=str, default=None, help='path of a saved_checkpoint')
    parser.add_argument('--train', dest='train', type=int, default=0, help='0 to test the model, 1 to train the model')
    parser.add_argument('--save-interval', dest='save_interval', type=int, default=10, help='epochs after which save the model')
    return parser.parse_args()


def save_model(model, epoch, optimizer, loss, path):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss


def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)

def image_warp(im,flow,device):
	"""Performs a backward warp of an image using the predicted flow.
	Args:
		im: Batch of images. [num_batch, channels, height, width]
		flow: Batch of flow vectors. [num_batch, 2, height, width]
	Returns:
		warped: transformed image of the same shape as the input image.
	"""

	## may think of swaping image axes
	cuda0 = torch.device('cuda:0')
	im = im.permute(0,2,3,1) ## [num_batch, height, width, channels]
	flow = flow.permute(0,2,3,1)
	num_batch, height, width, channels = im.size()
	max_x = int(width - 1)
	max_y = int(height - 1)
	zero = torch.zeros([],dtype=torch.int32).to(device).long()

	im_flat = torch.reshape(im,(-1,channels))
	flow_flat = torch.reshape(flow,(-1,2))

	flow_floor = torch.floor(flow_flat).long()
	bilinear_weights = flow_flat - torch.floor(flow_flat)

	pos_x = torch.arange(width).repeat(height*num_batch).long()
	grid_y = torch.arange(height).unsqueeze(1).repeat(1,width).long()
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
	batch_offsets = torch.arange(num_batch) * dim1
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


# def conv(c_in, c_out, K, S, P=None, d=None , activations=nn.ReLU(), batchnorm=True):
    
        # layers_list = 
        # layer = nn.Sequential(





