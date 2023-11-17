import os
import glob
import shutil
import random

def split(src_folder, dst_folder):
    # Cleaning out previous splits
    if os.path.isdir(dst_folder):
        shutil.rmtree(dst_folder)
    
    os.makedirs(os.path.join(dst_folder, "train", "images"))
    os.makedirs(os.path.join(dst_folder, "train", "masks"))

    os.makedirs(os.path.join(dst_folder, "test", "images"))
    os.makedirs(os.path.join(dst_folder, "test", "masks"))

    # Then split the dataset in a stratified way
    for folder in glob.glob(os.path.join(src_folder, "*")):
        print("folder: ", folder)
        imgs = glob.glob(os.path.join(folder, "images", "*"))
        print(imgs)

        random.seed(428)
        random.shuffle(imgs)
        
        train = imgs[:int(0.8*len(imgs))]
        test = imgs[int(0.8*len(imgs)):]

        for img in train:
            img_fn = os.path.basename(img)
            shutil.copy(img, os.path.join(dst_folder, "train", "images", img_fn))
            
            mask = img.replace("images", "masks")
            shutil.copy(mask, os.path.join(dst_folder, "train", "masks", img_fn))
        
        for img in test:
            img_fn = os.path.basename(img)
            shutil.copy(img, os.path.join(dst_folder, "test", "images", img_fn))
            
            mask = img.replace("images", "masks")
            shutil.copy(mask, os.path.join(dst_folder, "test", "masks", img_fn))

    return


src_folder = "/home/alvin/CIID/Datasets/ai_ready_datasets/separate"
dst_folder = "/home/alvin/CIID/Datasets/ai_ready_datasets/combined"
split(src_folder, dst_folder)