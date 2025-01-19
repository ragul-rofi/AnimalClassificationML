#Importing necessary library

import os
import shutil
from sklearn.model_selection import train_test_split

#Difining dataset paths

original_dataset_dir = "dataset"  #Original folder
base_dir = "new_dataset" #New folder for processed dataset
os.makedirs(base_dir, exist_ok = True)

#Splitting of the dataset folders

for split in ['train','validate','test']:
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir,exist_ok=True)

#Splitting each class folder

for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir,class_name)
    if os.path.isdir(class_path):
        images = [os.path.join(class_path, fname) for fname in os.listdir(class_path)]

        #Splitting ratio (train:validate:test) = 80:10:10
        train_images, temp_images = train_test_split(images, test_size=0.2,random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

        #Copying images to their split directories
        for split, split_images in zip(['train','validation','test'], [train_images, val_images,test_images]):
            split_class_dir = os.path.join(base_dir,split,class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in split_images:
                shutil.copy(img, os.path.join(split_class_dir,os.path.basename(img)))

print("Dataset successfully splitted")