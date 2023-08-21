# try to cover unknown with pytesseract
# cover unknown by eye 18 images
import os
import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
from random import randint


def show_img(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_QR1(image):
    a = randint(0, 3)
    for i in range(a):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    detector = cv2.QRCodeDetector()
    data, vertices, _ = detector.detectAndDecode(image)

    if data:
        return data
    else:
        return 'Unknown'


def read_QR2(image):
    a = randint(0, 3)
    for i in range(a):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    decoded_objects = decode(gray_image, symbols=[ZBarSymbol.QRCODE])

    if len(decoded_objects) != 1:   # if no or more than expected QRs found
        return 'Unknown'
    else:
        obj = decoded_objects[0]
        QR_code = obj.data.decode('utf-8')

        return QR_code


def cover_unknown(path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_paths = [path+name 
                    for name in os.listdir(path)
                        if 'Unknown' in name and any(name.lower().endswith(ext) for ext in image_extensions)]
    for unknown_img_path in image_paths:
        unknown_img = cv2.imread(unknown_img_path)
        unknown_img_right_side = unknown_img[:, unknown_img.shape[1]//2:]

        QR_code = read_QR1(unknown_img_right_side)
        if QR_code != 'Unknown':
            show_img(unknown_img_right_side, QR_code)
            print(unknown_img_path.replace('Unknown', str(QR_code)))
            cv2.imwrite(unknown_img_path.replace('Unknown', str(QR_code)), unknown_img)
            os.remove(unknown_img_path)
            print('saved')
            continue

        QR_code = read_QR2(unknown_img_right_side)
        if QR_code != 'Unknown':
            show_img(unknown_img_right_side, QR_code)
            print(unknown_img_path.replace('Unknown', str(QR_code)))
            cv2.imwrite(unknown_img_path.replace('Unknown', str(QR_code)), unknown_img)
            os.remove(unknown_img_path)
            print('saved')
            continue


for i in range(10):
    print(i)
    cover_unknown("/home/alperenlcr/Code/Plant_Grow_Tracking/Gabrofil/data/")