import pandas as pd
import os
import shutil
import numpy as np

# Run this script with: "sudo -E python3 split_folder.py"

# CHANGE TO YOUR COMBINED DATASET DIRECTORY
img_dir = '/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/'
df = pd.read_csv('/media/jer/data2/state-farm-distracted-driver-detection/driver_imgs_list.csv')

shuffle = False


people = df['subject']
classes = df['classname']
img_names = df['img']
num_samples = img_names.shape[0]

# if shuffle:
#     for i, img_name in enumerate(img_names):
#         rand_float = np.random.uniform(0, 1)
#         if rand_float <= 0.8:
#             shutil.copy(img_dir + img_name, img_dir + "shuffled_train/" + img_name)
#             print("copying to shuffled train " + img_name, int(people[i][1:]))
#         elif rand_float > 0.8:
#             shutil.copy(img_dir + img_name, img_dir + "shuffled_test/" + img_name)
#             print("copying to shuffled test " + img_name, int(people[i][1:]))    

# elif not shuffle:
#     for i, img_name in enumerate(img_names):
#         if int(people[i][1:]) <= 56:
#             shutil.copy(img_dir + img_name, img_dir + "train/" + img_name)
#             print("copying to train " + img_name, int(people[i][1:]))
#         elif int(people[i][1:]) > 56:
#             shutil.copy(img_dir + img_name, img_dir + "test/" + img_name)
#             print("copying to test " + img_name, int(people[i][1:]))
            
        
if not shuffle:
    for i, img_name in enumerate(img_names):
        rand_float = np.random.uniform(0, 1)
        if (int(people[i][1:]) <= 56) and ((classes[i] != "c0" and (rand_float <= 0.111)) or (classes[i] == "c0")):
            shutil.copy(img_dir + img_name, img_dir + "binary_train/" + img_name)
            print("copying to train " + img_name, int(people[i][1:]))
        # elif int(people[i][1:]) > 56:
        if (int(people[i][1:]) > 56) and ((classes[i] != "c0" and (rand_float <= 0.111)) or (classes[i] == "c0")):
            shutil.copy(img_dir + img_name, img_dir + "binary_test/" + img_name)
            print("copying to test " + img_name, int(people[i][1:]))
            
num_train = len(os.listdir(img_dir + "/train/"))
num_test = len(os.listdir(img_dir + "/test/"))

print("train ratio: ", num_train / num_samples)
print("test ratio: ", num_test / num_samples)
        
        