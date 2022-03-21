import numpy as np
import torch
import pandas as pd
import os
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision

# data is in /media/jer/data2/state-farm-distracted-driver-detection/imgs
# columns of csv are person[0], class name[1], and image name[2]

class DriverDataset(Dataset):
    def __init__(self):
        # CHANGE TO YOUR DIRECTORY
        self.img_dir = '/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/'
        self.df = pd.read_csv('/media/jer/data2/state-farm-distracted-driver-detection/driver_imgs_list.csv')
             
        self.people = self.df['subject']
        self.classes = self.df['classname']
        self.img_names = self.df['img']
        self.num_samples = self.img_names.shape[0]
      
    def __getitem__(self, index):
        return self.people[index], self.classes[index], self.img_names[index]
    
    def split_dataset(self, cutoff):
        dataset_1 = (self.people[0:cutoff], self.classes[0:cutoff], self.img_names[0:cutoff])
        dataset_2 = (self.people[cutoff:], self.classes[cutoff:], self.img_names[cutoff:])       
        return dataset_1, dataset_2
    
    def show_image(self, i):
        name = str(self.df.iloc[i, 2])
        directory = self.img_dir + name
        img = io.imread(directory)
        plt.imshow(img)
        plt.show()
        
    def make_batch(self, data, n):
        batch = np.zeros((n, 480, 640, 3), dtype='uint8')
        # picking n random rows of the dataset
        batch_indices = np.random.choice(data[2].shape[0], n, replace=False)
        
        for index, random_index in enumerate(batch_indices):
            print(data[2][random_index])
            batch[index] = plt.imread(self.img_dir + data[2][random_index])
        return batch

if __name__ == "__main__":          
    dataset = DriverDataset()
    # splitting dataset right before p061 starts (80%)... couldn't find labels for their test set
    train_data, test_data = dataset.split_dataset(17777)
    training_batch = dataset.make_batch(train_data, 10)

    for img in training_batch:
        plt.imshow(img)
        plt.show()

    # print(dataset[17777])
    # dataset.show_image(17777)
    # plt.show()

    ''' TODO:
    - cross validation - split training into training and validation
    '''