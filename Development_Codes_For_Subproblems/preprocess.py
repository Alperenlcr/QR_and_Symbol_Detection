import cv2
from pyzbar.pyzbar import decode
import os
from pytesseract import pytesseract


class Preprocess():
    def __init__(self, PATH_TO_FOLDER, QR_codes):
        self.main_path = PATH_TO_FOLDER
        self.Corrupted_path, self.QRs_path = self.create_adjacent_folders(QR_codes) # Creates Corrupteds, QRs folders and in QRS creates folder for QRcodes if they don't exist
        self.paths = self.get_paths()   # returns all the image paths according the phone
        self.main(QR_codes)


    def main(self, QR_codes):
        # counter and relation dictionaries are for avoiding long names for images
        # and relations dictionary has info needed
        counter = {}
        relations = {}
        for code in QR_codes:
            counter[code] = 0
            relations[code] = ['non']
        relations['c'] = ['non']
        counter['c'] = 0


        for folder_path in self.paths.keys():
            for image_path in self.paths[folder_path]:
                image = cv2.imread(image_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                decoded_objects = decode(gray_image)

                if len(decoded_objects) != 1:   # if no or more than expected QRs found
                    counter['c'] += 1
                    relations['c'].append(image_path)
                    p = f"{self.Corrupted_path}/{counter['c']}{'-'}{obj.rect.top}.{image_path.split('.')[-1]}"
                    print(p)
                    cv2.imwrite(p, image)
                    #print("{} : {} QR Codes Detected in a single photo!".format(image_path, len(decoded_objects)))
                else:
                    obj = decoded_objects[0]
                    # Rotate the image if its wrong
                    if obj.rect.left < image.shape[1]//2: # width
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
                        #print(image_path, "Turn the photo to right")
                    elif obj.rect.top > image.shape[0]//2:    # height
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        #print(image_path, "Turn the photo to left")

                    QRcode = obj.data.decode('utf-8') 

                    counter[QRcode] += 1
                    relations[QRcode].append(image_path)
                    p = f"{self.QRs_path}/{QRcode}/{counter[QRcode]}{'-'}{obj.rect.top}.{image_path.split('.')[-1]}"
                    print(p)
                    cv2.imwrite(p, image)


    def create_adjacent_folders(self, subfolder_names):
        base_folder = os.path.abspath(self.main_path)
        
        QRs_folder = os.path.join(os.path.dirname(base_folder), 'QRs')
        Corrupteds_folder = os.path.join(os.path.dirname(base_folder), 'Corrupteds')
        
        # Create the 'QRs' and 'Corrupteds' folders only if they don't exist
        if not os.path.exists(QRs_folder):
            os.makedirs(QRs_folder)
        
        if not os.path.exists(Corrupteds_folder):
            os.makedirs(Corrupteds_folder)

        subfolders = [os.path.join(QRs_folder, name) for name in subfolder_names]
        for subfolder in subfolders:
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

        return Corrupteds_folder, QRs_folder


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


# takes two params : Path to main images folder and QRcode ids
C = Preprocess('/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/Slike_poskus_25.7.2023/', ['533424', '289562', '774740', '705548', '579519', '301948'])
