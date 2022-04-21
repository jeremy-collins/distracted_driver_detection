"""
Script with Pytorch's dataloader class
"""

import glob
import os
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import csv
import pandas as pd


class ImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        # CHANGE TO YOUR DIRECTORY
        # self.img_dir = '../imgs/train/combined'
        self.img_dir = '/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/'

        # self.df = pd.read_csv('../driver_imgs_list.csv')
        self.df = pd.read_csv('/media/jer/data2/state-farm-distracted-driver-detection/driver_imgs_list.csv')

        # # colab dirs
        # self.img_dir = './combined/'
        # self.df = pd.read_csv('./driver_imgs_list.csv')

        
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(self.img_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(self.img_dir, self.test_folder)

        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(
        self, class_labels: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        """Fetches all (image path,label) pairs in the dataset.

        Args:
            class_labels: the class labels dictionary, with keys being the classes in this dataset
        Returns:
            list[(filepath, int)]: a list of filepaths and their class indices
        """

        img_paths = []  # a list of (filename, class index)

        ############################################################################
        # Student code begin
        ############################################################################
        
        self.img_names = self.df['img']
        self.classes = self.df['classname']
        
        img_dict = dict(zip(self.img_names, self.classes))
        

        # folder_list = os.listdir(self.curr_folder)
        # for folder in folder_list:
        #     for img in os.listdir(os.path.join(self.curr_folder, folder)):
        #         img_paths.append((os.path.join(self.curr_folder, folder, img), class_labels[folder]))
                
        for img in os.listdir(self.curr_folder):
            img_paths.append((self.curr_folder + '/' + img, int(img_dict[img][-1])))
        
        # for idx, img in enumerate(self.img_names):
        #     os.join(self.img_dir, img)
        #     # filename, class index
        #     img_paths.append((os.path.join(self.img_dir, img), class_labels[self.classes[idx]]))
            
        # print(img_paths)
        
        
        ############################################################################
        # Student code end
        ############################################################################

        return img_paths

    def get_classes(self) -> Dict[str, int]:
        """Get the classes (which are folder names in self.curr_folder)

        NOTE: Please make sure that your classes are sorted in alphabetical order
        i.e. if your classes are ['apple', 'giraffe', 'elephant', 'cat'], the
        class labels dictionary should be:
        {"apple": 0, "cat": 1, "elephant": 2, "giraffe":3}

        If you fail to do so, you will most likely fail the accuracy
        tests on Gradescope

        Returns:
            Dict of class names (string) to integer labels
        """

        classes = dict()
        ############################################################################
        # Student code begin
        ############################################################################

        # folder_names = os.listdir(self.curr_folder)
        # folder_names = sorted(folder_names)
        # classes = {os.path.basename(folder_names[i]):i for i in range(len(folder_names))}
        
        classes = {str(i):i for i in range(10)}
        
        ############################################################################
        # Student code end
        ############################################################################
        return classes

    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None
        ############################################################################
        # Student code begin
        ############################################################################

        img = Image.open(path).convert('L')
        
        ############################################################################
        # Student code end
        ############################################################################
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """
        img = None
        class_idx = None

        ############################################################################
        # Student code start
        ############################################################################
        # self.dataset = [(filepath, class_idx)]
        
        img = self.load_img_from_path(self.dataset[index][0])
        class_idx = self.dataset[index][1]
        
        img = self.transform(img)

        ############################################################################
        # Student code end
        ############################################################################
        return img, class_idx

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        l = 0

        ############################################################################
        # Student code start
        ############################################################################

        l = len(self.dataset)
                
        ############################################################################
        # Student code end
        ############################################################################
        return l