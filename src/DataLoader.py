from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import glob

class KITTIDataset(Dataset):
    def __init__(self, folder_name, transform = None, diff_frames = 1):
        self.folder_name = folder_name
        self.transform = transform
        self.num_seq = 200
        self.diff_frames = diff_frames
        self.num_frames = 21 - self.diff_frames

    def __len__(self):
        return self.num_seq * (self.num_frames)
    
    def __getitem__(self, index):
        frame_id = index % self.num_frames 
        seq_id = int(index / self.num_frames)

        if "testing" in self.folder_name:
            if(seq_id == 26):
                frame_id = frame_id % (16 - self.diff_frames)
            elif(seq_id == 167):
                frame_id = frame_id % (15 - self.diff_frames)
    
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

        if(frame_id + self.diff_frames < 10):
            frame2_framename = "0"
        else:
            frame2_framename = ""

        frame2_filename = self.folder_name + frame1_seqname + str(seq_id) + \
                            "_" + frame2_framename + str(frame_id+self.diff_frames) + ".png"

        
        # print(frame1_filename)
        # print(frame2_filename, "\n")

        frame1 = (Image.open(frame1_filename).convert('YCbCr').split()[0])
        frame2 = (Image.open(frame2_filename).convert('YCbCr').split()[0])
        # frame1 = np.array(Image.open(frame1_filename).convert('YCbCr').split()[0])
        # frame2 = np.array(Image.open(frame2_filename).convert('YCbCr').split()[0])

        sample = {'frame1':frame1, 'frame2':frame2}
        # sample = np.stack((frame1, frame2), axis=0)
        # imgplot = plt.imshow(frame1)
        # plt.show()
        # imgplot = plt.imshow(frame2)
        # plt.show()

        if(self.transform):
            sample = self.transform(sample)

        return sample


class MCLVDataset(Dataset):
    def __init__(self, folder_name, transform = None, diff_frames = 1):
        self.folder_name = folder_name
        self.transform = transform
        self.num_seq = 10
        self.diff_frames = diff_frames
        self.num_frames = 149 - self.diff_frames

        self.video_folders = glob.glob(folder_name +"*/")

        print(self.video_folders)
    def __len__(self):
        return self.num_seq * (self.num_frames)
    
    def __getitem__(self, index):
        frame_id = index % self.num_frames 
        seq_id = int(index / self.num_frames)

        
        frame1_filename = self.video_folders[seq_id] + str(frame_id) + ".png"
        frame2_filename = self.video_folders[seq_id] + str(frame_id + self.diff_frames) + ".png"

        # print(frame1_filename)
        # print(frame2_filename, "\n")


        frame1 = (Image.open(frame1_filename).convert('YCbCr').split()[0])
        frame2 = (Image.open(frame2_filename).convert('YCbCr').split()[0])

        sample = {'frame1':frame1, 'frame2':frame2}
        
        # sample = np.stack((frame1, frame2), axis=0)
        # imgplot = plt.imshow(frame1)
        # plt.show()
        # imgplot = plt.imshow(frame2)
        # plt.show()

        if(self.transform):
            sample = self.transform(sample)

        return sample

class KITTIStereoDataset(Dataset):
    def __init__(self, folder_name_1, folder_name_2, transform = None):
        self.folder_name_1 = folder_name_1
        self.folder_name_2 = folder_name_2
        self.transform = transform
        self.num_seq = 200
        self.num_frames = 21

    def __len__(self):
        return self.num_seq * (self.num_frames)
    
    def __getitem__(self, index):
        frame_id = index % self.num_frames 
        seq_id = int(index / self.num_frames)

        if "testing" in self.folder_name_1:
            if(seq_id == 26):
                frame_id = frame_id % 16
            elif(seq_id == 167):
                frame_id = frame_id % 15
    
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
        
        frame1_filename = self.folder_name_1 + frame1_seqname + str(seq_id) + \
                            "_" + frame1_framename + str(frame_id) + ".png"

        if(frame_id < 10):
            frame2_framename = "0"
        else:
            frame2_framename = ""

        frame2_filename = self.folder_name_2 + frame1_seqname + str(seq_id) + \
                            "_" + frame2_framename + str(frame_id) + ".png"

        
        # print(frame1_filename)
        # print(frame2_filename, "\n")

        frame1 = (Image.open(frame1_filename).convert('YCbCr').split()[0])
        frame2 = (Image.open(frame2_filename).convert('YCbCr').split()[0])
        # frame1 = np.array(Image.open(frame1_filename).convert('YCbCr').split()[0])
        # frame2 = np.array(Image.open(frame2_filename).convert('YCbCr').split()[0])

        sample = {'frame1':frame1, 'frame2':frame2}
        
        # sample = np.stack((frame1, frame2), axis=0)
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
        # frame1, frame2 = sample['frame1'], sample['frame2']

        # frame1 = (frame1/127.5) - 1 
        # frame2 = (frame2/127.5) - 1
        
        sample = sample/255.0
        # sample = (sample/127.5) - 1
        return sample
        # return {'frame1':frame1, 'frame2':frame2}
        

class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)
        # return {'frame1':torch.from_numpy(sample['frame1']), 
                # 'frame2':torch.from_numpy(sample['frame2'])}


class RandomCrop(object):
    def __init__(self, size):
        self.new_h = size[0]
        self.new_w = size[1]
    def __call__(self, sample):
        frame1, frame2 = sample['frame1'], sample['frame2']
        sample = np.stack((np.array(frame1), np.array(frame2)), axis=0)
        # h, w = frame1.shape[:2]
#         print("img shape ", sample.shape)
        h,w = sample.shape[1:]
        # print("h,w ",h,w)
        # print("new_h,new_w ",self.new_h,self.new_w)
        try:
            top = np.random.randint(0, h - self.new_h + 1)
            left = np.random.randint(0, w - self.new_w + 1)
        except:
            print("err shape ", sample.shape)

        sample = sample[:, top: top + self.new_h,
                        left: left + self.new_w]
        
        # frame1 = frame1[top: top + self.new_h,
                        # left: left + self.new_w]

        # frame2 = frame2[top: top + self.new_h,
                        # left: left + self.new_w]
        
        return sample
        # return {'frame1': frame1, 'frame2': frame2}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        frame1, frame2 = sample['frame1'], sample['frame2']
        if random.random() < self.p:
            frame1 = transforms.functional.vflip(frame1)
            frame2 = transforms.functional.vflip(frame2)
        return {'frame1': frame1, 'frame2': frame2}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        frame1, frame2 = sample['frame1'], sample['frame2']
        if random.random() < self.p:
            frame1 = transforms.functional.hflip(frame1)
            frame2 = transforms.functional.hflip(frame2)
        
        return {'frame1': frame1, 'frame2': frame2}


def main():
    # dataset = KITTIDataset(folder_name='../data_scene_flow_multiview/training/image_2/',
    # transform=transforms.Compose([RandomVerticalFlip(), 
    #     RandomHorizontalFlip(), 
    #     RandomCrop([320, 896]),
    #     Normalize(),
    #     ToTensor()
    # ]),
    # diff_frames = 2)

    # dataset = KITTIStereoDataset(folder_name_1='../data_scene_flow_multiview/training/image_2/',
    # folder_name_2='../data_scene_flow_multiview/training/image_3/',
    # transform=transforms.Compose([RandomVerticalFlip(), 
    #     RandomHorizontalFlip(), 
    #     RandomCrop([320, 896]),
    #     Normalize(),
    #     ToTensor()
    # ]))
    dataset = MCLVDataset(folder_name="/home/suhail/DL/let_it_flow/data/MCL-V/video_bitstream/",
    transform=transforms.Compose([RandomVerticalFlip(), 
        RandomHorizontalFlip(), 
        RandomCrop([320, 896]),
        Normalize(),
        ToTensor()
    ]), diff_frames=2)

    sample = dataset[0]

    dataloader = DataLoader(dataset, batch_size = 20, shuffle = True, num_workers = 4)

    for batch_ndx, sample in enumerate(dataloader):
        print(sample.shape)
if __name__ == "__main__":
    main()
