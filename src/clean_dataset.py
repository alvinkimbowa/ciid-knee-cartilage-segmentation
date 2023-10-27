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

    for img_path in tqdm(img_paths):
        image_save_fn = os.path.basename(img_path)
        image_save_path = f'{dst_image_folder}/{image_save_fn}'
        
        # Automatically obtain the corresponding mask
        mask = [msk for msk in mask_paths if image_save_fn.split('.')[0] in msk][0]
        mask_save_fn = image_save_fn.replace('.', '_mask.')
        mask_save_path = f'{dst_mask_folder}/{mask_save_fn}'
        
        # Skip those images that were already cleaned
        if image_save_fn in os.listdir(dst_image_folder):
            continue
        
        print("img_path: ", img_path)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        
        image, mask = clean_image(image, mask)

        cv2.imwrite(image_save_path, image)
        cv2.imwrite(mask_save_path, mask)
    
    return None


if __name__ == "__main__":
    folders = [
        # '1_Images_segmentations_matt', # Need to convert roi to segmentation masks
        '2_Control_Cartilage_Images_Segmentations_Ilker',
        '3_ACL_Cartilage_Images_Segmentations_Ilker', # Already clean
        # '4_Knee_Manual_Segmented_Images',
        '5.1_clarius_auto_msu_parmar_segmentations',
        '5.2_clarius_std_msu_parmar_segmentations',
        '5.3_ge_auto_msu_parmar_segmentations',
        '5.4_ge_std_msu_parmar_segmentations',
        'exteded field of view prajna', # To look into how to use this
        'Other anatomy',
    ]

    raw_dataset_folder = "../../Datasets/raw_datasets"
    ai_ready_dataset_folder = "../../Datasets/ai_ready_datasets_inpainted"
    for folder in folders:
        print("folder: ", folder)
        image_folder = os.path.join(raw_dataset_folder, folder, "images")
        mask_folder = os.path.join(raw_dataset_folder, folder, "masks")
        dst_image_folder = os.path.join(ai_ready_dataset_folder, folder, "images")
        dst_mask_folder = os.path.join(ai_ready_dataset_folder, folder, "masks")

        clean_images(image_folder, dst_image_folder, mask_folder, dst_mask_folder)