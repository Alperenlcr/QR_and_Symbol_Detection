# Figure how to find XXX then do

import glob
import os
import cv2
import numpy as np
import random
from pytesseract import pytesseract
from datetime import datetime, timedelta
import json
import pandas as pd

def show_img(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
#    cv2.destroyAllWindows()


def find_image_paths(path):
        image_paths = []
        image_extensions = ['jpg', 'jpeg', 'png', 'gif']
        for ext in image_extensions:
            search_pattern = os.path.join(path, '**', f'*.{ext}')
            image_paths.extend(glob.glob(search_pattern, recursive=True))

        return image_paths


def display_hsv_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    img = image[:, :image.shape[1]//2]
    height, width, channels = img.shape
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    show_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    show_img(dst, "sharpened_image")
    

def crop_convert_gray_save(image_path):
    image = cv2.imread(image_path)
    img = image[:, :image.shape[1]//2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image_path.replace('data', 'Images'), gray)


def crop_black_part(image_path):
    image = cv2.imread(image_path, 0)

    i = 1
    while len([[True] for x in image[-i] if x < 60]) > 20:
        i += 1
    cv2.imwrite(image_path.replace('Images', 'x'), image[:-i])

def matcher(image_path, template_path):
    from matplotlib import pyplot as plt
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img2 = img.copy()
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED'] # works bad: 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR' , cv2.TM_CCORR_NORMED
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)

        image = img2[top_left[1]-25:bottom_right[1]+20, top_left[0]-20:bottom_right[0]+20]
        # show_img(image)
        cv2.imwrite(image_path.replace('/x/', '/'+meth+'/'), image)


def remove_date(img_path):
    image = cv2.imread(img_path, 0)
    details = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng')
    x = 0
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
    image = image[:, x + width+20:]
    image = cv2.resize(image, (400, 150))
    cv2.imwrite(img_path.replace('yyy', 'zzz'), image)

def enhance_image(img_path):
    image = cv2.imread(img_path, 0)
    ratio = 0
    x = 80
    #show_img(image)
    original = image.copy()
    while ratio < 14 and x < 200:
        image[original <= x] = 0
        image[original > x] = 255
        count_lower_than_x = np.sum(image < x)
        ratio = (count_lower_than_x/(400*150))*100
        x += 2

    if ratio > 17:
        print(ratio, x)
        show_img(original)
    #show_img(image)

    cv2.imwrite(img_path.replace('zzz', 'ttt'), image)


def cropping_resizing_inverting(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)
    # crop
    # TOP
    i = 1
    while len([[True] for x in image[i] if x == 255]) < 20:
        i += 1
    if i > 15:
        i -= 15
    elif i > 10:
        i -= 10
    elif i > 5:
        i -= 5
    image = image[i:]

    # BOTTOM
    i = 1
    while len([[True] for x in image[-i] if x == 255]) < 40:
        i += 1
    image = image[:-i-10]
    
    # RIGHT
    i = 1
    # show_img(image.T)
    while len([[True] for x in image.T[-i] if x == 255]) < 20:
        i += 1
    image = image.T[:-i+10].T.T.T

    # LEFT
    image = np.rot90(image)
    i = 1
    while len([[True] for x in image[-i] if x == 255]) < 20:
        i += 1
    image = image[:-i+10]
    image = np.rot90(image)
    image = np.rot90(image)
    image = np.rot90(image)

    # image = cv2.resize(image, (105, 30))
    cv2.imwrite(image_path.replace('ttt', 'uuu'), image)



def make_bolder(image_path):
    # Load the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform image dilation to make white regions bolder
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    # Display the dilated image
    cv2.imwrite(image_path, dilated_image)



def invert_noise_crop(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original = image.copy()
    image[original <= 128] = 0
    image[original > 128] = 255
    width = len(image[0])
    height = len(image)
    threshold = width // 3
    
    for row in range(height):
        count_white = 0
        for col in range(width):
            if image[row, col] == 255:
                count_white += 1
                if count_white >= threshold:
                    image[row, col - threshold + 1 : col + 1] = 0
            else:
                count_white = 0
    
    top = 0
    while width == np.sum(image[top] == 0):
        top += 1
    bottom = 1
    while width == np.sum(image[-bottom] == 0):
        bottom += 1

    image = image[top:-bottom, :]        
#    show_img(image)
    cv2.imwrite(img_path, image)


def recognise_symbols(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original = image.copy()
    image[original <= 128] = 0
    image[original > 128] = 255
    width = len(image[0])
    height = len(image)
    threshold = (height * 8.5) // 10
    
    name = list('xxx')
    for col in range(width):
        count_white = 0
        for row in range(height):
            if image[row, col] == 255:
                count_white += 1
                if count_white > threshold:
                    cv2.line(image, (col, 0), (col, height-1), 128, 1)
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
    cv2.imwrite(img_path.replace('xxx', name).replace('uuu', 'result'), image)


def filter_duplicate_entries(data_list, time_threshold_sec=0):
    filtered_data = {
        "QR_code": [],
        "light": [],
        "date": [],
        "phone": [],
        "name": []
    }
    seen_entries = {}
    image_name = {}
    for entry in data_list:
        qr_code = entry["QR_code"]
        light = entry["light"]
        phone = entry["phone"]
        date_time = entry["date"]


        key = (qr_code, light, phone)
        if key in seen_entries:
            prev_date_time = seen_entries[key]
            if date_time - prev_date_time >= timedelta(seconds=time_threshold_sec):
                seen_entries[key] = date_time
                image_name[key] = entry["name"]
        else:
            seen_entries[key] = date_time
            image_name[key] = entry["name"]

    sorted_data = dict(sorted(seen_entries.items(), key=lambda item: item[1]))
    print(len(sorted_data))
    for i in sorted_data:
        filtered_data['QR_code'].append(i[0])
        filtered_data['light'].append(i[1])
        filtered_data['date'].append(seen_entries[i])
        filtered_data['phone'].append(i[2])
        filtered_data['name'].append(image_name[i])
    # Create a DataFrame
    df = pd.DataFrame(filtered_data)

    # Display the DataFrame as a table
    print(df)
    return filtered_data

# Function to filter rows with close dates
def filter_rows(group):
    prev_date = group.iloc[0]["date"]
    filtered_rows = [prev_date]
    
    for _, row in group.iterrows():
        if row["date"] - prev_date > timedelta(seconds=30):
            filtered_rows.append(row["date"])
        prev_date = row["date"]
    
    return group[group["date"].isin(filtered_rows)]


def create_json(image_paths):
    data = {
        "date": [],
        "QR_code": [],
        "light": [],
        "phone": [],
        "name": []
    }
    for info in image_paths:
        QR_code, light, date, phone, name = info.split('-')
        data['date'].append(datetime.strptime(date, "%Y:%m:%d %H:%M:%S"))
        data['QR_code'].append(QR_code.split('/')[-1])
        data['light'].append(light)
        data['phone'].append(phone)
        data['name'].append(name)

    df = pd.DataFrame(data).sort_values(by=["date"]).reset_index(drop=True)
    df.to_csv('/'.join(image_paths[0].split('/')[:-1]) + '/result.csv')
    print(df.head(50))
    filtered_df = df.groupby(["QR_code", "light", "phone"]).apply(filter_rows)

    #print(filtered_df)
    exit(1)
    filtered = filter_duplicate_entries(data_list)
    dicti = {}
    for q in QR_codes:
        dicti[q] = []
    for info in filtered:
        dicti[info["QR_code"]].append()
    # Specify the path to save the JSON file
    output_path = '/'.join(image_paths[0].split('/')[:-1]) + '/result.json'
    # Save the dictionary as a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(filtered, json_file, indent=4)


img_paths = find_image_paths('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/result/')
random.shuffle(img_paths)

template_paths = find_image_paths('/home/alperenlcr/Downloads/trainingSet/trainingSet/0/')
template_images = [cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) for template_path in template_paths]
for img_path in img_paths:
    # recognise_symbols(img_path)
    create_json(img_paths)
exit(1)

def find_and_draw_black_rectangle(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create a binary image
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Perform morphological opening to remove noise
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # Find contours in the opened image
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the leftmost and topmost points of the black rectangle
    leftmost_x = float('inf')
    topmost_y = float('inf')
    rectangle_found = False

    # Loop through the contours to find the most left and most top black rectangle
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width >= 300 and height >= 100:
            if x < leftmost_x and y < topmost_y:
                leftmost_x = x
                topmost_y = y
                rectangle_found = True

    # Draw the gray rectangle if a suitable black rectangle is found
    if rectangle_found:
        gray_color = (192, 192, 192)  # Gray color in BGR
        cv2.rectangle(image, (leftmost_x, topmost_y), (leftmost_x + 300, topmost_y + 100), gray_color, -1)

    # Display the image with the drawn rectangle
    cv2.imshow('Image with Drawn Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






image_paths = find_image_paths('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/data/')
original_images_paths = []
for image_path in image_paths:
    original_images_paths.append(f'/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/Slike_poskus_25.7.2023/{image_path.split("/")[-1].split("-")[-2]}/{image_path.split("/")[-1].split("-")[-1]}')
    print(original_images_paths[-1])
    find_and_draw_black_rectangle(original_images_paths[-1])







exit(1)
# delete the ones taken next to each with same qrcode

from config import QR_codes
from datetime import datetime


def convert_date_to_minute(time_string):
    # Parse the time string into a datetime object
    time_format = '%Y:%m:%d %H:%M:%S'
    time_datetime = datetime.strptime(time_string, time_format)

    # Calculate the total minutes
    total_minutes = time_datetime.hour * 60 + time_datetime.minute
    return total_minutes

def filter_images(path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_paths = [path+name 
                    for name in os.listdir(path)
                        if any(name.lower().endswith(ext) for ext in image_extensions)]
    info = {}
    for qrcode in QR_codes:
        info[qrcode] = {}

    for image_path in image_paths:
        code = image_path.split('/')[-1].split('-')[0]
        date = convert_date_to_minute(image_path.split('/')[-1].split('-')[2])
        info[code][date] = image_path

    for key in info.keys():
        print(key, ':')
        for key2 in info[key].keys():
            print(f'    {key2} : {info[key][key2]}')


filter_images("/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/data/")










exit(1)

def binarization():
    # Load the color image
    image_path = '/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/data/579519-xxx-2023:07:25 10:27:28-Redmi Note 11 Pro-IMG_20230725_102728.jpg'
    color_image = cv2.imread(image_path)
    color_image = color_image[:, :color_image.shape[1] // 2]

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Display the original and equalized images side by side
    stacked_images = cv2.hconcat([gray_image, equalized_image])
    cv2.imshow('Original vs Equalized', stacked_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Apply thresholding to create a binary image
    threshold_value = 110  # Adjust this threshold value as needed
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Display the binary image
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('/home/alperenlcr/Code/Plant_Grow_Tracking/binary_image.jpg',binary_image)

def detect_date(path='/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/data/'):

    method = cv2.TM_SQDIFF_NORMED

    # Read the images from the file
    small_image = cv2.imread('/home/alperenlcr/Code/Plant_Grow_Tracking/dateB.jpg', cv2.IMREAD_GRAYSCALE)

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_paths = [path+name 
                    for name in os.listdir(path)
                        if any(name.lower().endswith(ext) for ext in image_extensions)]

    for image_path in image_paths:
        large_image = cv2.imread(image_path)
        large_image = large_image[:, :large_image.shape[1]//2]
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Display the original and equalized images side by side
        stacked_images = cv2.hconcat([gray_image, equalized_image])
        result = cv2.matchTemplate(small_image, stacked_images, method)

        # We want the minimum squared difference
        mn,_,mnLoc,_ = cv2.minMaxLoc(result)

        # Draw the rectangle:
        # Extract the coordinates of our best match
        MPx,MPy = mnLoc

        # Step 2: Get the size of the template. This is the same size as the match.
        trows,tcols = small_image.shape[:2]

        # Step 3: Draw the rectangle on large_image
        cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

        # Display the original image with the rectangle around the match.
        cv2.imshow('output',large_image)

        # The image is only displayed if we call this
        cv2.waitKey(0)



#detect_date()

def detect_and_draw_blue_lines(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = image[:, :image.shape[1]//2]
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define a lower and upper threshold for blue color in BGR
    lower_blue = np.array([10, 20, 30], dtype=np.uint8)
    upper_blue = np.array([50, 60, 70], dtype=np.uint8)

    # Create a binary mask for blue pixels
    blue_mask = cv2.inRange(image, lower_blue, upper_blue)

    # Find the x-coordinate of the leftmost blue pixel
    leftmost_blue_pixel_indices = np.where(blue_mask == 255)
    if leftmost_blue_pixel_indices[1].size > 0:
        leftmost_blue_x = np.min(leftmost_blue_pixel_indices[1])
    else:
        leftmost_blue_x = 0

    # Find the y-coordinate of the topmost blue pixel
    topmost_blue_pixel_indices = np.where(blue_mask == 255)
    if topmost_blue_pixel_indices[0].size > 0:
        topmost_blue_y = np.min(topmost_blue_pixel_indices[0])
    else:
        topmost_blue_y = 0

    # Draw lines for both leftmost and topmost blue pixels
    line_color = (0, 255, 0)  # Green color for the lines
    line_thickness = 2
    image_with_lines = image.copy()
    # image_with_lines = cv2.line(image_with_lines, (leftmost_blue_x, 0), (leftmost_blue_x, image.shape[0]), line_color, line_thickness)
    # image_with_lines = cv2.line(image_with_lines, (0, topmost_blue_y), (image.shape[1], topmost_blue_y), line_color, line_thickness)

    # Draw a red circle at the intersection point
    intersection_point = (leftmost_blue_x - 30, topmost_blue_y - 30)
    circle_radius = 3
    circle_color = (0, 0, 255)  # Red color for the circle
    cv2.circle(image_with_lines, intersection_point, circle_radius, circle_color, -1)

    # Display the image with the blue lines and circle
    cv2.imshow('Image with Blue Lines and Circle', image_with_lines)#[topmost_blue_y - 30:topmost_blue_y + 70, leftmost_blue_x - 30:leftmost_blue_x + 270])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the image path
path='/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/data/'
image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
image_paths = [path+name 
                for name in os.listdir(path)
                    if any(name.lower().endswith(ext) for ext in image_extensions)]
for image_path in image_paths:
    print(image_path)
    detect_and_draw_blue_lines(image_path)