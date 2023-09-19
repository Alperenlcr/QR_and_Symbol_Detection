import os
import glob
import numpy as np
import cv2
from dbr import BarcodeReader, BarcodeReaderError
from kraken import binarization
from PIL import Image as IMG
from PIL.ExifTags import TAGS
from qreader import QReader
from tkinter import *
import pandas as pd
from scipy import spatial
from pyzbar.pyzbar import decode, ZBarSymbol
from math import sqrt
import concurrent.futures

QR_codes = ['533424', '289562', '774740', '705548', '579519', '301948']
image_extensions = ['jpg', 'jpeg', 'png', 'gif']

RAW_IMAGES_PATH = '<PATH>/Phones_25/'
ALL_DATA_CSV_PATH = '<PATH>/all_data.csv'
DATA_CSV_PATH = '<PATH>/data.csv'
SAVED_IMGS_PATH = '<PATH>/Saved_imgs/'

column_names = ['date', 'QRcode', 'experiment_number', 'light', 'phone_name', 'image_name', 'black_rectangle_corners', 'height_phone_b', 'height_phone_t', 'height_phone', 'circles_info', 'angle_multiplier_info', 'savingID'] # circles_info is according to normalized tubes image
values = ['image_name', 'experiment_number', 'light', 'QRcode']

save_normalized_tubes = True
save_normalized_tubes_with_circles = False
