import os
import glob
import json
import shutil
import cv2

dataset_json = {
    "channel_names": {  # formerly modalities
      "0": "nonCT",
    },
    "labels": {  # THIS IS DIFFERENT NOW!
      "background": 0,
      "cartilage": 1,
    },
    "file_ending": ".png"
  }


def convert_dataset(src_folder, dst_folder):
  if os.path.isdir(dst_folder):
    shutil.rmtree(dst_folder)
  
  os.makedirs(dst_folder)
  os.makedirs(f"{dst_folder}/imagesTr")
  os.makedirs(f"{dst_folder}/labelsTr")
  
  for folder in glob.glob(f"{src_folder}/*"):
    # Always create a fresh version of the nnunet dataset

    if not os.path.isdir(folder):
      continue
    
    imgs = glob.glob(f"{folder}/images/*.png")
    masks = glob.glob(f"{folder}/masks/*.png")
    for i, (img, mask) in enumerate(zip(imgs, masks)):
      print("img: ", img)
      print("mask: ", mask)
      img_fn = os.path.basename(img)
      img_fn = img_fn.replace(".png", f"_{i:03d}_0000.png")
      shutil.copy(img, f"{dst_folder}/imagesTr/{img_fn}")
      
      msk_fn = os.path.basename(mask)
      msk_fn = msk_fn.replace(".png", f"_{i:03d}.png")
      mask_image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
      cv2.imwrite(f"{dst_folder}/labelsTr/{msk_fn}", mask_image/255)

  dataset_json["numTraining"] = len(os.listdir(f"{dst_folder}/imagesTr"))
  with open(f"{dst_folder}/dataset.json", "w") as f:
    json.dump(dataset_json, f)


if __name__=="__main__":
  dataset = "/home/alvin/CIID/Datasets/ai_ready_datasets"
  dst_folder = "/home/alvin/CIID/nnUNetv2/nnUNetv2_raw/Dataset030_combined_knee_ultrasound"
  convert_dataset(src_folder=dataset, dst_folder=dst_folder)