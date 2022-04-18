import numpy as np
import torch
import pandas as pd
import os
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from typing import Sequence, Tuple

# data is in /media/jer/data2/state-farm-distracted-driver-detection/imgs
# columns of csv are person[0], class name[1], and image name[2]

class DriverDataset(Dataset):
    def __init__(self, shuffle=True):
        # CHANGE TO YOUR DIRECTORY
        self.img_dir = '../imgs/train/combined'
        # self.img_dir = '/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/'
        self.df = pd.read_csv('../driver_imgs_list.csv')
        # self.df = pd.read_csv('/media/jer/data2/state-farm-distracted-driver-detection/driver_imgs_list.csv')


        # shuffle the dataframe
        if (shuffle):
            self.df = self.df.sample(frac=1)

        self.people = self.df['subject']
        self.classes = self.df['classname']
        self.img_names = self.df['img']
        self.num_samples = self.img_names.shape[0]

        self.width = 64
        self.height = 48

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
        new_width = self.width
        new_height = self.height
        if (gray):
            batch = np.zeros((n, new_height, new_width), dtype='float')
        else:
            batch = np.zeros((n, new_height, new_width, 3), dtype='float')
        labels = np.empty((n), dtype='int')
        people = np.empty((n), dtype='int')
        # picking n random rows of the dataset
        batch_indices = np.random.choice(data[2].index.to_numpy(), n, replace=False)

        for index, random_index in enumerate(batch_indices):
            # print(data[2][random_index])
            img = Image.open(os.path.join(self.img_dir, data[2][random_index]))

            img = img.resize((new_width, new_height))
            # img = self.augment(img, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, flip=False, crop=True, angle=10, normalize=False)
            img = self.augment(img, normalize=True)

            # plt.imshow(img)
            # plt.show()


            if (gray):
                img = ImageOps.grayscale(img)

            if (gray and edges):
                img = img.filter(ImageFilter.FIND_EDGES)
            batch[index] = np.moveaxis(np.asarray(transforms.ToTensor()(img)), [0,1,2], [2,0,1])
            people[index] = int(data[0][random_index][1:])
            labels[index] = int(data[1][random_index][1:])

            # plt.imshow(batch[index])
            # plt.show()

        return batch, people, labels

    def k_folds(self, X, y, kfold):

        x_split = np.split(X, kfold, axis=0)
        y_split = np.split(y, kfold, axis=0)

        # for i in range(kfold):
        #     x_tuple = x_split[:i] + x_split[(i+1):]
        #     y_tuple = y_split[:i] + y_split[(i+1):]

        #     x_curr = np.concatenate(x_tuple)
        #     y_curr = np.concatenate(y_tuple)

        #     weight = self.ridge_fit_closed(x_curr, y_curr, c_lambda)
        #     prediction = self.predict(x_split[i], weight)
        #     errors[i] = self.rmse(prediction, y_split[i])

        print(x_split)
        return x_split, y_split

    def augment(
        self,
        image,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
        flip=False,
        crop=False,
        angle=0,
        normalize=False,
        mean=None,
        std=None):

        transform = transforms.Compose(
        [
            # transforms.Resize(size),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.RandomRotation(angle),
        ]
        )

        image = transform(image)

        if flip:
            image = transforms.RandomHorizontalFlip()(image)

        if crop:
            # would need to increase image resolution before cropping
            image = transforms.RandomCrop((self.height, self.width))(image)

        if normalize:
            ind1 = image.dim() - 2
            ind2 = image.dim() - 1
            # image = transforms.Normalize(image.mean(dim=[ind1,ind2]), image.std(dim=[ind1,ind2]))(image)
            image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        image = transforms.ToPILImage()(image)

        return image


if __name__ == "__main__":
    dataset = DriverDataset()
    # splitting dataset right before p061 starts (80%)... couldn't find labels for their test set
    train_data, test_data = dataset.split_dataset(17777)
    training_batch, train_people, train_labels = dataset.make_batch(train_data, 10, gray=False, edges=False)
    testing_batch, test_people, test_labels = dataset.make_batch(test_data, 10, gray=False, edges=False)

    x_split, y_split = dataset.k_folds(training_batch, testing_batch, 5)
    # (person, class, image_name)
    print("x split: ", len(x_split))
    print(x_split[0].shape)
    print("y split: ", len(y_split))
    print(y_split[0].shape)


    for img in training_batch:
        print(type(img))
        plt.imshow(img)
        plt.show()

    # print(dataset[17777])
    # dataset.show_image(17777)
    # plt.show()
