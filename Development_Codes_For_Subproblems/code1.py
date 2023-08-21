# Create 'data' named folder
# travel to every folder and read images. do operations on them and save them in format as
#     apply qr detection
#     crop them
#     'QRcode-XXX-date-phone_name-image_name'

import cv2
from pyzbar.pyzbar import decode
import os
from config import *
from PIL import Image
from PIL.ExifTags import TAGS


class StepOne():
    def __init__(self, PATH_TO_FOLDER, QR_codes):
        self.QR_codes = QR_codes
        self.main_path = PATH_TO_FOLDER
        self.data_path = self.create_adjacent_folders()
        self.paths = self.get_paths()   # returns all the image paths according the phone
        self.cropped_paths = self.get_cropped_paths()   # returns all the image paths according the phone
        self.unknown_images = [x.split('data/Unknown-')[1][4:-4] for x in self.cropped_paths if 'Unknown' in x]


    def gather_info_and_save_cropped_images(self):
        for folder_path in self.paths.keys():
            for image_path in self.paths[folder_path]:
                image_name = image_path.split('/')[-1].split('.')[0]
                phone_name = image_path.split('/')[-2]
                date = self.get_original_capture_date(image_path)
                
                if (f'{date}-{phone_name}-{image_name}' not in self.unknown_images): # or ('Honor' not in phone_name):
                    continue
                QRcode, x, image = self.read_QR(image_path)
                light = 'xxx'

                image = image[100:400+int(x), :]
                file_name = f'{QRcode}-{light}-{date}-{phone_name}-{image_name}.jpg'
                print(self.data_path+file_name)
                cv2.imwrite(self.data_path+file_name, image)


    def read_QR(self, image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray_image)

        if len(decoded_objects) != 1:   # if no or more than expected QRs found
            return 'Unknown', 800, image
        else:
            obj = decoded_objects[0]
            QR_code = obj.data.decode('utf-8')
            coordinate = obj.rect.top
            # Rotate the image if its wrong
            if obj.rect.left < image.shape[1]//2: # width
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
            elif obj.rect.top > image.shape[0]//2:    # height
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                return QR_code, coordinate, image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            decoded_objects = decode(gray_image)
            if len(decoded_objects) != 1:
                return QR_code, 500, image

            return QR_code, decoded_objects[0].rect.top, image


    def get_original_capture_date(self, image_path):
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == "DateTimeOriginal":
                        return value
        except Exception as e:
            print("Error:", e)
        return None


    def create_adjacent_folders(self):
        base_folder = os.path.abspath(self.main_path)
        
        data_folder = os.path.join(os.path.dirname(base_folder), 'data/')
        
        # Create the 'data' if they don't exist
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        return data_folder


    def get_paths(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        paths = {}
        # find the names of subfolders
        sub_folders = [self.main_path+name+'/'
                            for name in os.listdir(self.main_path) 
                                if os.path.isdir(os.path.join(self.main_path, name))]

        # get the list of images and create a dict as folder_path:[images_names]
        for sub_folder in sub_folders:
            paths[sub_folder] = [sub_folder+name 
                                    for name in os.listdir(sub_folder)
                                        if any(name.lower().endswith(ext) for ext in image_extensions)]

        return paths


    def get_cropped_paths(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        return [self.data_path+name 
                                    for name in os.listdir(self.data_path)
                                        if any(name.lower().endswith(ext) for ext in image_extensions)]


# takes two params : Path to main images folder and QRcode ids
C = StepOne(RAW_IMAGES_PATH, QR_codes)
C.gather_info_and_save_cropped_images()  # it saves every image in format of 'QRcode-XXX-date-image_phone-image_name'
