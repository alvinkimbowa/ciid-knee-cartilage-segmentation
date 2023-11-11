import os
import glob
import json
import shutil
import cv2


def convert_dataset(results_folder):
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
  

if __name__=="__main__":
  results_folder = "/home/alvin/CIID/nnUNetv2/nnUNet_results"
  convert_dataset(results_folder)