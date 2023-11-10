import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.remove_text_from_image import remove_text


def crop_image(image, mask):
    # Obtain the extreme verticle x coordinates of the mask
    c, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = c[0]
    left = tuple(c[c[:, :, 0].argmin()][0])[0]
    right = tuple(c[c[:, :, 0].argmax()][0])[0]
    
    cropped_image = image[:, left:right]
    cropped_mask = mask[:, left:right]

    return (cropped_image, cropped_mask)


def clean_image(image, mask=None):
    '''
        This function takes in an image, crops it basing on the segmentation 
        mask and removes text from the image
    '''
    if mask is not None:
        # Crop image
        image, mask = crop_image(image, mask)

    # Remove text
    image = remove_text(image)

    return (image, mask)


def clean_images(image_folder, dst_image_folder, mask_folder=None, dst_mask_folder=None):
    img_paths = glob.glob(f"{image_folder}/*")
    mask_paths = glob.glob(f"{mask_folder}/*")

    # Ensure the destination folders exist
    if not os.path.isdir(dst_image_folder):
        os.makedirs(dst_image_folder)
    if mask_folder and not os.path.isdir(dst_mask_folder):
        os.makedirs(dst_mask_folder)

    for i,img_path in enumerate(tqdm(img_paths)):
        image_save_fn = os.path.basename(img_path)
        image_save_path = f'{dst_image_folder}/{image_save_fn}'
        
        # Automatically obtain the corresponding mask
        mask = [msk for msk in mask_paths if image_save_fn.split('.')[0] in msk][0]
        mask_save_fn = image_save_fn
        mask_save_path = f'{dst_mask_folder}/{mask_save_fn}'
        
        # Skip those images that were already cleaned
        if image_save_fn in os.listdir(dst_image_folder):
            continue
        
        print("img_path: ", img_path)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        
        if not "rutgers_health" in image_folder:
            image, mask = clean_image(image, mask)

        cv2.imwrite(image_save_path, image)
        cv2.imwrite(mask_save_path, mask)
    
    return None


if __name__ == "__main__":
    raw_dataset_folder = "../../Datasets/raw_datasets"
    ai_ready_dataset_folder = "../../Datasets/ai_ready_datasets"
    datasets = ["msu", "rutgers"]
    for folder in os.listdir(raw_dataset_folder):
        if folder.split('_')[0] not in datasets:
            continue
        print("folder: ", folder)
        image_folder = os.path.join(raw_dataset_folder, folder, "images")
        mask_folder = os.path.join(raw_dataset_folder, folder, "masks")
        dst_image_folder = os.path.join(ai_ready_dataset_folder, folder, "images")
        dst_mask_folder = os.path.join(ai_ready_dataset_folder, folder, "masks")

        clean_images(image_folder, dst_image_folder, mask_folder, dst_mask_folder)