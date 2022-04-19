import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    # folder_list = os.listdir(dir_name)
    # means = np.array([])
    # stdevs = np.array([])
    # avg = 0
    # stdev = 0
    
    dataset = np.array([], dtype="uint8")
    count = 0
    
    # for folder in folder_list:
    #     for img in os.listdir(os.path.join(dir_name, folder)):
    for dirpath, dirnames, img_list in os.walk(dir_name):
        for img in img_list:
            if not dirnames:
                img_path = os.path.join(dirpath, img)
                img = Image.open(img_path)
                img = img.convert('L')
                img = np.array(img)
                dataset = np.append(dataset, img.reshape((-1,)))
                count += 1

    mean = np.mean(dataset)/255
    std = np.std(dataset)/255
    
    print(mean, std)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
