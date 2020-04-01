import argparse
import pdb
import os,sys
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lrd', dest='lr_disc', type=float, default=1e-3, help='learning rate_for_discriminator')
    parser.add_argument('--lrg', dest='lr_gen', type=float, default=1e-3, help='learning rate_for_generator')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--checkpoint', dest='checkpoint_path', type=str, default=None, help='path of a saved_checkpoint')
    parser.add_argument('--train', dest='train', type=int, default=0, help='0 to test the model, 1 to train the model')
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
        if not os.path.exists():
            os.mkdir(path)


