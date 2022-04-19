import pandas as pd
import os
import shutil

img_dir = '/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/'
df = pd.read_csv('/media/jer/data2/state-farm-distracted-driver-detection/driver_imgs_list.csv')

people = df['subject']
classes = df['classname']
img_names = df['img']
num_samples = img_names.shape[0]


for i, img_name in enumerate(img_names):
    if int(people[i][1:]) <= 56:
        shutil.copy("/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/" + img_name, "/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/train/" + img_name)
        print("copying to train " + img_name)
        print("train", int(people[i][1:]))
    elif int(people[i][1:]) > 56:
        # shutil.copy("/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/" + img_name, "/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/combined/test/" + img_name)
        print("not copying to test " + img_name)
        print("test", int(people[i][1:]))

        
        