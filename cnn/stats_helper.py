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
    
    img_size = (640, 480)
    dataset = np.array([], dtype="uint8")
    count = 0
    sum = np.array([0], dtype="int64")
    sq_sum = np.array([0], dtype="int64")
    
    
    # for folder in folder_list:
    #     for img in os.listdir(os.path.join(dir_name, folder)):
    for dirpath, dirnames, img_list in os.walk(dir_name):
        for img in img_list:
            if not dirnames:
                img_path = os.path.join(dirpath, img)
                img = Image.open(img_path)
                img = img.convert('L')
                img = np.array(img, dtype="uint8")
                sum += np.sum(img)
                sq_sum += np.sum(np.square(img))
                # dataset = np.append(dataset, img.reshape((-1,)))
                count += 1
                
                
    print(count, " images processed")

    count = count * img_size[0] * img_size[1] 
    sum = sum / 255
    sq_sum = sq_sum / 255
                
    mean = sum / count
    var = (sq_sum / count) - (mean ** 2) 
    print("var: ", var)
    std = np.sqrt(var)          



    # mean = np.mean(dataset)/255
    # std = np.std(dataset)/255
    
    # mean is 0.36668006
    print("mean: ", mean)
    
    # std is 0.51483374
    print("standard deviation: ", std)
    
    ############################################################################
    # Student code end
    ############################################################################
    return mean[0], std[0]
