import numpy as np
import torch
import pandas as pd
import os
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image, ImageOps, ImageFilter

# data is in /media/jer/data2/state-farm-distracted-driver-detection/imgs
# columns of csv are person[0], class name[1], and image name[2]

class DriverDataset(Dataset):
    def __init__(self, shuffle=True):
        # CHANGE TO YOUR DIRECTORY
        self.img_dir = '../imgs/train/combined'
        self.df = pd.read_csv('../driver_imgs_list.csv')

        # shuffle the dataframe
        self.df = self.df.sample(frac=1)

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

    def make_batch(self, data, n, gray=True, edges=True):
        new_width = 64
        new_height = 48
        if (gray):
            batch = np.zeros((n, new_height, new_width), dtype='uint8')
        else:
            batch = np.zeros((n, new_height, new_width, 3), dtype='uint8')
        labels = np.empty((n), dtype='int')
        people = np.empty((n), dtype='int')
        # picking n random rows of the dataset
        batch_indices = np.random.choice(data[2].index.to_numpy(), n, replace=False)

        for index, random_index in enumerate(batch_indices):
            # print(data[2][random_index])
            img = Image.open(os.path.join(self.img_dir, data[2][random_index]))
            img = img.resize((new_width, new_height))
            if (gray):
                img = ImageOps.grayscale(img)

            if (gray and edges):
                img = img.filter(ImageFilter.FIND_EDGES)
            batch[index] = np.asarray(img)
            people[index] = int(data[0][random_index][1:])
            labels[index] = int(data[1][random_index][1:])
        return batch, people, labels

if __name__ == "__main__":
    dataset = DriverDataset()
    # splitting dataset right before p061 starts (80%)... couldn't find labels for their test set
    train_data, test_data = dataset.split_dataset(17777)
    training_batch, train_people, train_labels = dataset.make_batch(train_data, 10)

    for img in training_batch:
        plt.imshow(img)
        plt.show()

    # print(dataset[17777])
    # dataset.show_image(17777)
    # plt.show()

    ''' TODO:
    - cross validation - split training into training and validation
    '''