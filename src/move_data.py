from glob import glob
import os
import random
import shutil

def main():
    data_path = "/home/suhail/DL/let_it_flow/data/MCL-V/video_bitstream/data_pairs/"
    train_path = "/home/suhail/DL/let_it_flow/data/train/"
    test_path = "/home/suhail/DL/let_it_flow/data/test/"
    
    if not os.path.isdir(train_path):
        os.mkdir(train_path)

    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    folders_name = glob(data_path+"*/")
    train_counter = 0
    test_counter = 0

    for folder_name in folders_name:
        file_names = glob(folder_name + "*.npz")
        for file_name in file_names:
            if(random.uniform(0,1) >= 0.85):
                new_file_name = test_path + str(test_counter) + ".npz"
                test_counter += 1
            else:
                new_file_name = train_path + str(train_counter) + ".npz"
                train_counter += 1
            shutil.move(file_name, new_file_name)

if __name__ == "__main__":
    main()