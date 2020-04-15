from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch

class KITTIDataset(Dataset):
    def __init__(self, folder_name, transform = None):
        self.folder_name = folder_name
        self.transform = transform
        self.num_seq = 200
        self.num_frames = 20
    def __len__(self):
        return self.num_seq * (self.num_frames)
    
    def __getitem__(self, index):
        frame_id = index % self.num_frames 
        seq_id = int(index / self.num_frames)

        if(seq_id < 10):
            frame1_seqname = "00000"
        elif(seq_id < 100):
            frame1_seqname = "0000"
        else:
            frame1_seqname = "000"

        if(frame_id < 10):
            frame1_framename = "0"
        else:
            frame1_framename = ""
        
        frame1_filename = self.folder_name + frame1_seqname + str(seq_id) + \
                            "_" + frame1_framename + str(frame_id) + ".png"

        if(frame_id + 1 < 10):
            frame2_framename = "0"
        else:
            frame2_framename = ""

        frame2_filename = self.folder_name + frame1_seqname + str(seq_id) + \
                            "_" + frame2_framename + str(frame_id+1) + ".png"


        frame1 = np.array(Image.open(frame1_filename).convert('YCbCr').split()[0])
        frame2 = np.array(Image.open(frame2_filename).convert('YCbCr').split()[0])

        sample = {'frame1':frame1, 'frame2':frame2}
        
        # imgplot = plt.imshow(frame1)
        # plt.show()
        # imgplot = plt.imshow(frame2)
        # plt.show()

        if(self.transform):
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        frame1, frame2 = sample['frame1'], sample['frame2']

        frame1 = (frame1/127.5) - 1 
        frame2 = (frame2/127.5) - 1

        return {'frame1':frame1, 'frame2':frame2}

class ToTensor(object):
    def __call__(self, sample):
        return {'frame1':torch.from_numpy(sample['frame1']), 
                'frame2':torch.from_numpy(sample['frame2'])}


class RandomCrop(object):
    def __init__(self, size):
        self.new_h = size[0]
        self.new_w = size[1]
    def __call__(self, sample):
        frame1, frame2 = sample['frame1'], sample['frame2']
        h, w = frame1.shape[:2]

        # print("h,w ",h,w)
        # print("new_h,new_w ",self.new_h,self.new_w)
        top = np.random.randint(0, h - self.new_h)
        left = np.random.randint(0, w - self.new_w)

        frame1 = frame1[top: top + self.new_h,
                        left: left + self.new_w]

        frame2 = frame2[top: top + self.new_h,
                        left: left + self.new_w]

        return {'frame1': frame1, 'frame2': frame2}

def main():
    dataset = KITTIDataset(folder_name='../data_scene_flow_multiview/training/image_2/',
    transform=transforms.Compose([RandomCrop([320, 896]),
        Normalize(),
        ToTensor()
    ]
    ))

    dataloader = DataLoader(dataset, batch_size = 20, shuffle = True, num_workers = 4)

    for batch_ndx, sample in enumerate(dataloader):
        print(sample['frame1'].shape)
if __name__ == "__main__":
    main()