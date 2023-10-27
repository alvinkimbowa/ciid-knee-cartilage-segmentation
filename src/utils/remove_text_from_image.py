import os
import cv2
import random
import keras_ocr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PIPELINE = keras_ocr.pipeline.Pipeline()


def compute_grad(image_gray, mode:str)->np.ndarray:
    """
    Function to compute the gradients magnitude of the image 
    set mode to "double" if you want to apply it two times
    """
    # Compute the gradients using the Sobel operator
    dx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude and direction of the gradients
    mag = np.sqrt(dx**2 + dy**2)
    
    if mode == "double":
        # Compute the gradients using the Sobel operator
        dx = cv2.Sobel(mag, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(mag, cv2.CV_64F, 0, 1, ksize=3)

        # Compute the magnitude and direction of the gradients
        magmag = np.sqrt(dx**2 + dy**2)
        return (magmag/magmag.max()*255).astype(np.uint8)
        
    return (mag/mag.max()*255).astype(np.uint8)


def detect_draw(pipeline, image_gray, viz):
    img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    #read image from the an image path (a jpg/png file or an image url)
    # Prediction_groups is a list of (word, box) tuples
    b = pipeline.recognize([img], detection_kwargs={"text_threshold":0.5})
    #print image with annotation and boxes
    if viz:
        plt.figure(num=1)
        plt.imshow(img)
        plt.show(block=False)
        keras_ocr.tools.drawAnnotations(image=img, predictions=b[0])

    return b


def patch_black(img, bb):
    for box in bb:
        box = np.array(box[1]).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [box], 0, (0,0,0),-1)
    
    return img


def inpaint_text(img, bb):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in bb[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        # Extract the patch within the bounding box from the original image
        patch = img[int(y0):int(y3), int(x0):int(x1)]
        
        # Some patches are too small (have zero dimension) they should be skipped
        if  not all(patch.shape):
            continue
        # plt.figure(num=2)
        # plt.imshow(patch, cmap='gray')
        # plt.show()

        # Threshold the grayscale image to create a binary mask of the black regions
        _, thresh = cv2.threshold(patch, 1, 255, cv2.THRESH_BINARY)

        # Update mask
        mask[int(y0):int(y3), int(x0):int(x1)] = thresh

    inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

    return inpainted_img


def remove_text(image_gray, pipeline=PIPELINE):
    mag = compute_grad(image_gray,"single")
    bb = detect_draw(pipeline, mag, viz=False)
    clean_image = patch_black(image_gray, bb[0])
    # clean_image = inpaint_text(image_gray, bb)
    return clean_image


if __name__== "__main__":
    PATH = "Datasets/raw_datasets/5.1_clarius_auto_msu_parmar_segmentations/images"
    files = os.listdir(PATH)

    # Choose random image from folder
    n = random.randint(0,len(files))
    image = cv2.imread(os.path.join(PATH, files[n]), cv2.IMREAD_GRAYSCALE)
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title("Original image")
    plt.show(block=False)

    clean_image = remove_text(image)

    # Visualize the images

    plt.figure()
    plt.imshow(clean_image, cmap='gray')
    plt.title("Original image")
    plt.show()
