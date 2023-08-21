import os
import glob
import numpy as np
import cv2
from kraken import binarization
from PIL import Image as IMG
from PIL.ExifTags import TAGS
from qreader import QReader
from tkinter import *
import pandas as pd


QR_codes = ['533424', '289562', '774740', '705548', '579519', '301948']
image_extensions = ['jpg', 'jpeg', 'png', 'gif']

RAW_IMAGES_PATH = '<PATH>/Gabrofil/Slike_poskus_25.7.2023/'
ALL_DATA_CSV_PATH = '<PATH>/all_data.csv'
DATA_CSV_PATH = '<PATH>/data.csv'

column_names = ['date', 'QRcode', 'experiment_number', 'light', 'phone_name', 'image_name']
values = ['image_name', 'experiment_number', 'light', 'QRcode']

