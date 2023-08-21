import glob
import cv2
import os
import numpy as np
import random
from pytesseract import pytesseract
from datetime import datetime, timedelta
import json
import pandas as pd
from dbr import BarcodeReader, EnumErrorCode
reader = BarcodeReader()


def find_image_paths(path):
        image_paths = []
        image_extensions = ['jpg', 'jpeg', 'png', 'gif']
        for ext in image_extensions:
            search_pattern = os.path.join(path, '**', f'*.{ext}')
            image_paths.extend(glob.glob(search_pattern, recursive=True))

        return image_paths

image_paths = find_image_paths('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/Slike_poskus_25.7.2023/')
random.shuffle(image_paths)
i = 0
s = []
for image_path in image_paths:
    i += 1
    print(i)
    results = reader.decode_file(image_path)
    if results is None:
        print("No codes found")
    else:
        for r in results:
            if r.barcode_text[-6:] not in s:
                s.append(r.barcode_text[-6:])
            print(r.barcode_text[-6:])
            print(r.localization_result.localization_points)
print(s)