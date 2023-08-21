import os
import numpy as np
from pytesseract import pytesseract
import cv2


class InfoExtractor():
    def __init__(self, PATH_TO_FOLDER, QR_codes):
        self.QRs_path = PATH_TO_FOLDER + 'QRs/'
        self.Corrupteds_path = PATH_TO_FOLDER + 'Corrupteds/'
        self.paths = self.get_paths_QRs()
        self.paths['c'] = self.get_paths_Corrupteds()
        self.main()


    def main(self):
        for folder in self.paths.keys():
            title = folder.split('/')[-2]
            #print(title)
            for img_path in self.paths[folder]:
                image = cv2.imread(img_path)
                image = image[:400+int(img_path.split('.')[0].split('-')[1]), :image.shape[1]//2]
                # It converts the BGR color space of image to HSV color space
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Threshold of blue in HSV space
                lower_blue = np.array([60, 35, 20])
                upper_blue = np.array([180, 255, 255])
            
                # preparing the mask to overlay
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                
                # The black region in the mask has the value of 0,
                # so when multiplied with original image removes all non-blue regions
                result = cv2.bitwise_and(image, image, mask = mask)
            
                cv2.imshow(title, result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Grayscale, Gaussian blur, Otsu's threshold
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                # Morph open to remove noise and invert image
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                invert = 255 - opening

#                data = pytesseract.image_to_string(invert, config='digits')
                data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')

                print(data)


    def get_paths_QRs(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        paths = {}
        # find the names of subfolders
        sub_folders = [self.QRs_path+name+'/'
                            for name in os.listdir(self.QRs_path) 
                                if os.path.isdir(os.path.join(self.QRs_path, name))]

        # get the list of images and create a dict as folder_path:[images_names]
        for sub_folder in sub_folders:
            paths[sub_folder] = [sub_folder+name 
                                    for name in os.listdir(sub_folder)
                                        if any(name.lower().endswith(ext) for ext in image_extensions)]

        return paths


    def get_paths_Corrupteds(self):
        return [self.Corrupteds_path+name
                    for name in os.listdir(self.Corrupteds_path)
                        if any(name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif'])]


# takes two params : Path to main images folder and QRcode ids
C = InfoExtractor('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/', ['533424', '289562', '774740', '705548', '579519', '301948'])
