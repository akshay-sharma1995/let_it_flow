import numpy as np
import glob 
import skvideo.io
import skvideo.utils
import skvideo.datasets
import skimage.io
import os
from matplotlib import pyplot as plt
import ipdb
import time

class DataPreProcessor():
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.file_names = glob.glob(self.folder_name + "*.mp4")
        # self.file_names = glob.glob(self.folder_name + "*.yuv")
        self.num_files = len(self.file_names)
        print(self.num_files)
    
    def savePair(self, filename, pair_num, frames_combined):
        if not os.path.isdir(filename):
            os.mkdir(filename)
        path = filename + '/' + str(pair_num)
        np.savez(path, frames_combined)

    def normalizeFrame(self, frame):
        frame = frame/127.5
        frame -= 1
        return frame

    def processVideo(self, idx):
        filename = self.file_names[idx]
        frames = skvideo.io.vread(filename, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
        # frames = skvideo.io.vread(filename, height=1080, width=1920, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
        print("Shape ", frames.shape)
        num_frames = frames.shape[0]
        for i in range(num_frames - 1):
            frame_1 = frames[i]
            # skimage.io.imshow(frame_1)
            # plt.show()
            frame_2 = frames[i+1]
            frame_1 = self.normalizeFrame(frame_1)
            frame_2 = self.normalizeFrame(frame_2)
            frames_combined = np.stack((frame_1, frame_2), axis = 2)
            self.savePair(filename[:-4], i, frames_combined)


def main():
    data_folder = "/home/akshay/Desktop/sem4/24789/project/dataset/MCL-V/temp/"
    # data_folder = "/home/akshay/Desktop/sem4/24789/project/dataset/MCL-V/reference_YUV_sequences/"
    preprocessor = DataPreProcessor(data_folder)
    # ipdb.set_trace()    
    for i in range(preprocessor.num_files):
        start_time = time.time()
        preprocessor.processVideo(i)
        print("time taken", time.time() - start_time)
        break

if __name__ == "__main__":
    main()
