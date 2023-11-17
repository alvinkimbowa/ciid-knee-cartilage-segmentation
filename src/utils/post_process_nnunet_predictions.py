import os
import glob
import json
import shutil
import cv2


def convert_results(results_folder):
  predictions = glob.glob(f"{results_folder}/*/*/*fold_*/validation/*.png")
  for prediction in predictions:
    print("prediction: ", prediction)

    src_folder, fn = os.path.split(prediction)
    dst_folder = src_folder + "_post_processed"
    save_path = os.path.join(dst_folder, fn)
    if not os.path.isdir(dst_folder):
      os.makedirs(dst_folder)

    pred_mask = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(save_path, pred_mask*255)
  

def rescale_images(folder, dst_folder):
  predictions = glob.glob(f"{folder}/*.png")
  for prediction in predictions:
    print("prediction: ", prediction)

    save_path = os.path.join(dst_folder, os.path.basename(prediction))
    if not os.path.isdir(dst_folder):
      os.makedirs(dst_folder)

    pred_mask = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(save_path, pred_mask*255)


if __name__=="__main__":
  # results_folder = "/home/alvin/CIID/nnUNetv2/nnUNet_results"
  # convert_results(results_folder)
  
  folder = "labelsTs"
  src_folder = f"/home/alvin/CIID/nnUNetv2/nnUNet_raw/Dataset030_combined_knee_ultrasound/{folder}"
  dst_folder = f"/home/alvin/CIID/nnUNetv2/nnUNet_raw/Dataset030_combined_knee_ultrasound/{folder}_postprocessed"
  rescale_images(src_folder, dst_folder)