import os
import glob
import numpy as np
import cv2
import random
from config import *
from PIL import Image
from PIL.ExifTags import TAGS
from pyzbar.pyzbar import decode, ZBarSymbol
from random import randint


class ExtractData():
    def __init__(self):
        self.create_folder(RESULT_PATH)


    # displays image
    def display(self, img, title='Image'):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Finds all images (including the ones in subfoler) in the given folder, returns their paths as list
    def find_image_paths(self, path) -> list:
        image_paths = []

        for ext in image_extensions:
            search_pattern = os.path.join(path, '**', f'*.{ext}')
            image_paths.extend(glob.glob(search_pattern, recursive=True))

        random.shuffle(image_paths)
        return image_paths


    # Creates folder with given path if allready no exits
    def create_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)


    def gray_to_binary_image(self, image, ratio_goal, x=80):
        ratio = 0
        original = image.copy()
        while ratio < ratio_goal*0.8 and x < 200:
            image[original <= x] = 0
            image[original > x] = 255
            count_lower_than_x = np.sum(image < x)
            ratio = (count_lower_than_x/(original.shape[0]*original.shape[1]))*100
            x += 2

        # if ratio > ratio_goal*1.2:
        #     print(ratio, x)
        return image


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


    def read_QR2(self, original, image):
        def calculate_new_coordinate(olds, oldW):
            coord_x, coord_y = olds
            new_x = coord_y
            new_y = oldW - coord_x
            return (new_x, new_y)

        for i in range(4):
            detector = cv2.QRCodeDetector()

            data, vertices, _ = detector.detectAndDecode(image)

            if data in QR_codes:
                x, y, w, h = cv2.boundingRect(vertices)
                center = (x + w // 2, y + h // 2)

                if i > 0:
                    print()
                for j in range(i):
                    center = calculate_new_coordinate(center, image.shape[1])
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                original = original[center[1]-200:center[1]+200, :original.shape[1]//2]
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                print(center, i)
                return data, original
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return 'Unknown', original


    def read_QR1(self, original, image):
        def calculate_new_coordinate(olds, oldW):
            coord_x, coord_y = olds
            new_x = coord_y
            new_y = oldW - coord_x
            return (new_x, new_y)

        for i in range(4):
            decoded_objects = decode(image, symbols=[ZBarSymbol.QRCODE])

            if len(decoded_objects) == 1:   # if no or more than expected QRs found
                obj = decoded_objects[0]
                QR_code = obj.data.decode('utf-8')
                center = (obj.rect.left + obj.rect.width // 2, obj.rect.top + obj.rect.height // 2)
                if i > 0:
                    print()
                for j in range(i):
                    center = calculate_new_coordinate(center, image.shape[1])
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                original = original[center[1]-200:center[1]+200, :original.shape[1]//2]
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                print(center, i)
                return QR_code, original
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return 'Unknown', original


    def rotate_name_crop(self):
        def save(image_path, modified_img):
            image_name = image_path.split('/')[-1].split('.')[0]
            phone_name = image_path.split('/')[-2]
            date = self.get_original_capture_date(image_path)
            light = 'xxx'
            file_name = f'{QR_code}-{light}-{date}-{phone_name}-{image_name}.jpg'
            cv2.imwrite(RESULT_PATH+file_name, modified_img)

        temp = self.find_image_paths(RAW_IMAGES_PATH)
        image_paths = [x for x in temp if ('Unknown4' in x)]
        read = []
        for image_path in image_paths:
            QR_code = str()
            for rsz in [x*0.5 for x in range(3, 20)]:
                image = cv2.imread(image_path)
                # Rotate if width > height:
                if image.shape[1] > image.shape[0]:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                image = cv2.resize(image, (int(image.shape[0]/rsz), int(image.shape[1]/rsz)))
                print(image.shape)
                img1 = image[:image.shape[0]//2, :]
                img2 = cv2.rotate(image[image.shape[0]//2:, :], cv2.ROTATE_180)
                imgs =  [img1, img2]
                for rg in range(20, 60, 5):
                    f = False
                    for i in range(2):
                        original = imgs[i].copy()
                        test_image = imgs[i].copy()

                        QR_code, _ = self.read_QR1(original, test_image)
                        if QR_code not in QR_codes:
                            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                            test_image = self.gray_to_binary_image(test_image, ratio_goal=rg)
                            QR_code, _ = self.read_QR2(original, test_image)
                        if QR_code in QR_codes:
                            save(image_path, _)
                            read.append(image_path)
                            f = True
                            break
                    if f:
                        break
            if f:
                break

                # QR_code, modified_img = self.read_QR1(image)
                # if QR_code in QR_codes:
                #     read.append(image_path)
                #     save(image_path, modified_img)
                #     continue

                # QR_code, modified_img = self.read_QR2(image)
                # if QR_code in QR_codes:
                #     read.append(image_path)
                #     save(image_path, modified_img)
                #     continue


        for image_path in  [x for x in image_paths if x not in read]:
            image_name = image_path.split('/')[-1].split('.')[0]
            phone_name = image_path.split('/')[-2]
            date = self.get_original_capture_date(image_path)
            light = 'xxx'
            file_name = f'Unknown-{light}-{date}-{phone_name}-{image_name}.jpg'
            cv2.imwrite(RESULT_PATH+file_name, cv2.imread(image_path))


S = ExtractData()

# this means that the function below never runned before
if len(S.find_image_paths(RESULT_PATH)) == 0:
    S.rotate_name_crop()
    if len([True for path in S.find_image_paths(RESULT_PATH) if 'Unknown' in path]) != 0:
        print('Make sure that the unknown images are in the correct way vertically. Correct them by hand if they are not.')
        inp = 'e'
        while inp != '1':
            inp = input('Then enter "1" to code to continue : ')
