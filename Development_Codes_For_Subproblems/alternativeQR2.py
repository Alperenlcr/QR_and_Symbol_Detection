from qreader import QReader
import cv2
import glob
import os
import random


# Create a QReader instance
qreader = QReader()

def read_QR(image_path):
    # Get the image that contains the QR code (QReader expects an uint8 numpy array)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Use the detect_and_decode function to get the decoded QR data
    decoded_text = qreader.detect_and_decode(image=image, return_bboxes=True)
    print(len(decoded_text))


def find_image_paths(path):
        image_paths = []
        image_extensions = ['jpg', 'jpeg', 'png', 'gif']
        for ext in image_extensions:
            search_pattern = os.path.join(path, '**', f'*.{ext}')
            image_paths.extend(glob.glob(search_pattern, recursive=True))

        return image_paths

read_QR('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/579519-xxx-2023:07:25 10:29:33-Honor Magic Vs-IMG_20230725_102933.jpg')
exit()
img_paths = find_image_paths('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/Slike_poskus_25.7.2023/')
random.shuffle(img_paths)
print(len(img_paths))

for image_path in img_paths:
    read_QR(image_path)
