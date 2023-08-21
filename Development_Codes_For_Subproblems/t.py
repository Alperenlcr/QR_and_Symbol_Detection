import random
import cv2
from kraken import binarization
from PIL import Image
import numpy as np
from pytesseract import pytesseract

import glob
import os

def solve(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    # Convert the OpenCV image (NumPy array) to a Pillow image
    pil_image = Image.fromarray(image)
    # Apply binarization using Kraken's binarization module
    binary_pil_image = binarization.nlbin(pil_image)  # Simple Niblack binarization

    # Convert the Pillow binary image back to a NumPy array for displaying
    binary_image = np.array(binary_pil_image)
    binary_image = binary_image[image.shape[0]//3:-image.shape[0]//6, :-8]

    # RIGHT
    right_black = 0    # Find the rightmost blackness
    bold = 5
    for i in range(binary_image.shape[1]):
        column = binary_image[:, i]
        if not np.all(column > 230):  # Check if the entire column is not white
            bold -= 1
            if bold <= 0:
                right_black = i
        else:
            bold = 5
    binary_image = binary_image[:, :right_black]
    # LEFT
    right_white_line = 0    # Find the rightmost white line
    bold = 5
    for i in range(binary_image.shape[1]):
        column = binary_image[:, i]
        if np.all(column > 230):  # Check if the entire column is white
            bold -= 1
            if bold <= 0:
                right_white_line = i
        else:
            bold = 5
    binary_image = binary_image[:, right_white_line+5:]

    # TOP
    i = 1
    while len([[True] for x in binary_image[i] if x < 25]) < 10:
        i += 1
    binary_image = binary_image[i:]

    # BOTTOM
    i = 1
    while len([[True] for x in binary_image[-i] if x < 25]) < 20:
        i += 1
    binary_image = binary_image[:-i-5]

    # Remove line remains
    width = binary_image.shape[1]
    height = binary_image.shape[0]
    threshold = width // 3
    for row in range(height):
        count_black = 0
        for col in range(width):
            if binary_image[row, col] < 25:
                count_black += 1
                if count_black >= threshold:
                    binary_image[row, col - threshold + 1 : col + 1] = 255
            else:
                count_black = 0

    # BOTTOM
    i = 1
    while len([[True] for x in binary_image[-i] if x < 25]) < 20:
        i += 1
    binary_image = binary_image[:-i-5]

    # Inverting
    binary_image = cv2.bitwise_not(binary_image)

    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Recognise
    original = binary_image.copy()
    binary_image[original <= 128] = 0
    binary_image[original > 128] = 255
    width = binary_image.shape[1]
    height = binary_image.shape[0]
    threshold = (height * 7) // 10
    
    name = list('xxx')
    for col in range(width):
        count_white = 0
        for row in range(height):
            if binary_image[row, col] == 255:
                count_white += 1
                if count_white > threshold:
                    cv2.line(binary_image, (col, 0), (col, height-1), 128, 1)
                    if col <= width//3:
                        name[0] = '1'
                    elif col >= (width//3)*2:
                        name[2] = '1'
                    elif col <= (width//3)*2 and col >= (width//3):
                        name[1] = '1'
            else:
                count_white = 0
    if name[1] == '1':
        name = list('xxx')
    else:
        for i in range(3):
            if name[i] == 'x':
                name[i] = '0'
    name = ''.join(name)

    cv2.imshow(name, binary_image)
    cv2.waitKey(0)
   # cv2.imwrite(image_path.replace('Result', 'test'), binary_image)
    return
    # Display the combined image



    details = pytesseract.image_to_data(binary_image, output_type=pytesseract.Output.DICT, lang='eng')
    x = 0
    ori = binary_image.copy()
    # Iterate through the detected words and their bounding box information
    for i, word in enumerate(details['text']):
        if 'ate' in word:  # Replace 'ate' with the word you're looking for
            x = details['left'][i]
            y = details['top'][i]
            width = details['width'][i]
            height = details['height'][i]
            f = 1
            # Draw a rectangle around the word 'ate'
            # cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Display the image with the drawn rectangle
            binary_image = binary_image[y-2*height:y+2*height, x + width+20:]

    # Display or save the binary image
# Get the heights of the two images
    height1, width1 = ori.shape[:2]
    height2, width2 = binary_image.shape[:2]

    # Calculate the padding needed for the smaller image
    padding = abs(height1 - height2) // 2

    # Create a blank white canvas with the larger height
    canvas_height = max(height1, height2)
    canvas_width = width1 + width2 + padding
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

    # Place the first image on the canvas
    canvas[:height1, :width1] = ori

    # Place the second image on the canvas with padding
    canvas[padding:padding + height2, width1:width1 + width2] = binary_image

    # Display the combined image
    cv2.imshow('Combined Image', canvas)
    cv2.waitKey(0)


def find_image_paths(path):
        image_paths = []
        image_extensions = ['jpg', 'jpeg', 'png', 'gif']
        for ext in image_extensions:
            search_pattern = os.path.join(path, '**', f'*.{ext}')
            image_paths.extend(glob.glob(search_pattern, recursive=True))

        return image_paths


image_paths = find_image_paths('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/Result/')
random.shuffle(image_paths)
solve('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/Result/579519-xxx-2023:07:25 10:44:05-Huawei P60 pro-IMG_20230725_104405.jpg')
for image_path in image_paths:
    solve(image_path)
