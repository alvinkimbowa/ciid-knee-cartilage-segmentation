import cv2
import numpy as np

def compare_images(a,b):
    '''
        The function takes in either an image or the path to the image and compares the two to check if they
        are the same.
    '''

    if isinstance(a, str):
        a = cv2.imread(a)
    if isinstance(b, str):
        b = cv2.imread(b)
    
    if a.shape != b.shape:
        print(f"Pictures are of different shapes: a - {a.shape}, b - {b.shape}")
        return
        
    difference = cv2.subtract(a, b)    
    result = not np.any(difference)
    if result is True:
        print("Pictures are the same")
    else:
        cv2.imwrite("ed.jpg", difference )
        print("Pictures are different, the difference is stored as ed.jpg")