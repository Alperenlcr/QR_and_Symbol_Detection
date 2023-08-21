import os
import numpy as np
from pytesseract import pytesseract
import cv2
import glob
import random


def find_image_paths(path):
        image_paths = []
        image_extensions = ['jpg', 'jpeg', 'png', 'gif']
        for ext in image_extensions:
            search_pattern = os.path.join(path, '**', f'*.{ext}')
            image_paths.extend(glob.glob(search_pattern, recursive=True))

        return image_paths

img_paths = find_image_paths('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/uuu/')
random.shuffle(img_paths)
i = 0
for img_path in img_paths:
    image = cv2.imread(img_path, 0)
    data = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    i += 1
    print(i, data)