import torch
import numpy as np
from torch.utills.data import Dataset, Dataloader
import os

class image_pair_dataset(Dataset):

  def __init__(self, root_dir, transform=None, channels_first=False)):

  # def __init__(self, root_dir, transform=None, channels_first=False):
  """
  Args:
  root_dir (string): Directory with all the image pairs
  transform (callable, optional): Optional transform to be applied on a sample
  channels_first (bool): Swap the channels axis to index 0
  """

    self.root_dir = root_dir
    self.transform = transform
    self.channels_first = channels_first

  def __len__(self):
    return len(self.data)

  def __get_item__(self,idx):
    file_name = os.path.join(self.root_dir, "{}.npy".format(idx)) 
    sample = np.load(file_name)['arr_0']
    if(self.transform):
      sample = self.transform(sample)
    if(self.channels_first):
      sample.permute(0,3,1,2)


    return sample
