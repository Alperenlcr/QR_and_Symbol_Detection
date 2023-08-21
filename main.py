from config import *


class Decode_QR_Symbols():
    def __init__(self):
        # Create a QReader instance
        self.qreader = QReader()
        self.df = pd.DataFrame(columns=column_names)
        self.main()


####### UTILS #######


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

        return image_paths


    # With given image path it returns date of creation
    def get_original_capture_date(self, image_path):
        try:
            with IMG.open(image_path) as img:
                exif_data = img._getexif()
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == "DateTimeOriginal":
                        return value
        except Exception as e:
            print("Error:", e)
        return None


    def find_longest_continuous_subarray_mid(self, arr):
        max_length = 0
        current_length = 0
        start_index = 0
        end_index = 0

        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1] + 1:
                current_length += 1
            else:
                current_length = 0

            if current_length > max_length:
                max_length = current_length
                start_index = i - current_length
                end_index = i

        longest_subarray = arr[start_index:end_index + 1]
        return longest_subarray[len(longest_subarray)//2]


    def find_longest_subarray_and_replace(self, arr):
        i = len(arr)-1
        count = 0
        while arr[i] < 50:
            arr[i] = 255
            count += 1
            i -= 1
        max_length = 0
        max_length_index = 0
        current_length = 0
        current_index = 0

        seen255 = False
        for i in range(len(arr)):
            if arr[i] == 0 and seen255:
                if current_length == 0:
                    current_index = i
                current_length += 1
                if current_length > max_length:
                    max_length = current_length
                    max_length_index = current_index
            elif arr[i] == 255:
                current_length = 0
                seen255 = True
            
        arr2 = [0]*len(arr)
        if max_length != 0:
            arr2[max_length_index:max_length_index+max_length] = [255] * max_length

        return arr2


####### END OF UTILS #######
####### READ QR #######


    # Get the image that contains the QR code (QReader expects an uint8 numpy array)
    def read_QR(self, gray_image):
        # Use the detect_and_decode function to get the decoded QR data
        decoded_text = self.qreader.detect_and_decode(image=gray_image, return_bboxes=True)
        if len(decoded_text) != 1:
            return 'Unknown', None

        bounding_box = {
                        'top_left': 
                            {
                                'x': decoded_text[0][0][0],
                                'y': decoded_text[0][0][1]
                            },
                        'bottom_right': 
                            {
                                'x': decoded_text[0][0][2],
                                'y': decoded_text[0][0][3]
                            }
                        }
        QR_code = decoded_text[0][1]

        if QR_code in QR_codes:
            return QR_code, bounding_box 
        return 'Unknown', None


    def info(self, image_path, QR_code):
        image_name = image_path.split('/')[-1].split('.')[0]
        phone_name = image_path.split('/')[-2]
        date = self.get_original_capture_date(image_path)
        light = 'xxx'
        file_name = f'{QR_code}-{light}-{date}-{phone_name}-{image_name}.jpg'
        return file_name


    def crop_image(self, gray_image, bounding_box):
        height = gray_image.shape[0]
        width = gray_image.shape[1]
        if bounding_box['bottom_right']['y'] > height//2:
            gray_image = cv2.rotate(gray_image, cv2.ROTATE_180)
            bounding_box_new = {
                                'top_left': 
                                    {
                                        'x': width - bounding_box['bottom_right']['x'],
                                        'y': height - bounding_box['bottom_right']['y']
                                    },
                                'bottom_right': 
                                    {
                                        'x': width - bounding_box['top_left']['x'],
                                        'y': height - bounding_box['top_left']['y']
                                    }
                                }
            bounding_box = bounding_box_new
        image = gray_image[bounding_box['top_left']['y']-50:bounding_box['bottom_right']['y']+100, :width//2]
        return image


    def rotate_name_crop(self, image_path):
        QR_code = str()
        rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # Rotate if width > height:
        if rgb_image.shape[1] > rgb_image.shape[0]:
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        QR_code, bounding_box = self.read_QR(rgb_image)

        info = self.info(image_path, QR_code)
        if QR_code in QR_codes:
            return info, self.crop_image(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), bounding_box)
        else:
            return info, cv2.imread(image_path)


####### END OF READ QR #######
####### HANDWRITING DIGIT RECOGNITION #######


    def crop_gray_to_binary(self, image):
        # Convert the OpenCV image (NumPy array) to a Pillow image
        pil_image = IMG.fromarray(image[:,:-8])
        # Apply binarization using Kraken's binarization module
        binary_pil_image = binarization.nlbin(pil_image)  # Simple Niblack binarization
        # Convert the Pillow binary image back to a NumPy array for displaying
        binary_image = np.array(binary_pil_image)
        # RIGHT
        mid = binary_image[binary_image.shape[0]//3:(binary_image.shape[0]*2)//3, :]
        right_black = 0    # Find the rightmost blackness
        bottom_index = 0
        bold = 5
        for i in range(mid.shape[1]):
            column = mid[:, i]
            if not np.all(column > 230):  # Check if the entire column is not white
                bold -= 1
                if bold <= 0:
                    right_black = i
                    bottom_index = np.where(column == 0)[0][-1] + binary_image.shape[0]//3
            else:
                bold = 5
        image = image[bottom_index-100:bottom_index+100, :right_black]
        
        # Convert the OpenCV image (NumPy array) to a Pillow image
        pil_image = IMG.fromarray(image)
        # Apply binarization using Kraken's binarization module
        binary_pil_image = binarization.nlbin(pil_image)  # Simple Niblack binarization
        # Convert the Pillow binary image back to a NumPy array for displaying
        binary_image = np.array(binary_pil_image)

        # LEFT
        right_white_line = 0    # Find the rightmost white line
        bold = 10
        for i in range(binary_image.shape[1]):
            column = binary_image[:, i]
            if np.all(column > 230):  # Check if the entire column is white
                bold -= 1
                if bold <= 0:
                    right_white_line = i
            else:
                bold = 10
        binary_image = binary_image[:, right_white_line+5:]
        # TOP
        i = 1
        while len([[True] for x in binary_image[i] if x < 25]) < 10:
            i += 1
        binary_image = binary_image[i:]
        
        # BOTTOM"
        i = 0
        while len([[True] for x in binary_image[i] if x > 200]) != binary_image.shape[1]:
            i += 1
        binary_image = binary_image[:i]

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

        # Inverting
        binary_image = cv2.bitwise_not(binary_image)

        # Make bolder
        kernel = np.ones((3, 3), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        ## Crop left and right black spaces for the last time
        mid = binary_image[binary_image.shape[0]//3:(binary_image.shape[0]*2)//3, :]
        # right crop point
        iR = mid.shape[1] - 1
        while np.all(mid[:, iR] < 20) and iR > 0:   # Check if the entire column is black
            iR -= 1
        if iR == 0:
            self.display(mid)
            raise Exception
        # left crop point
        iL = 0
        while np.all(mid[:, iL] < 20) and iL < mid.shape[1] - 1:   # Check if the entire column is black
            iL += 1
        if iL == mid.shape[1] - 1:
            self.display(mid)
            raise Exception

        if iR < mid.shape[1]-11:
            iR += 10
        elif iR < mid.shape[1]-6:
            iR += 5

        if iL > 10:
            iL -= 10
        elif iL > 5:
            iL -= 5

        binary_image = binary_image[:, iL:iR]

        return binary_image


    def detection(self, img):
        original = img.copy()
        img[original <= 128] = 0
        img[original > 128] = 255
        latest = 0
        increase = 0
        decrease = 0
        belowest_white = 0
        for i in range(0, len(img), 3):
            row = img[i]
            curr = np.count_nonzero(row)
            if curr != 0:
                belowest_white = i
                if curr > latest+1:
                    increase += 1
                    latest = curr
                    cv2.line(img, (0, i), (len(row)-1, i), 64, 1)
                elif curr < latest-1:
                    decrease += 1
                    latest = curr
                    cv2.line(img, (0, i), (len(row)-1, i), 192, 1)
        ratio = decrease/(decrease+increase)*100
        if ratio > 5 and belowest_white < (img.shape[0]*2)//3:
            ratio = 100


        if ratio < 14:
            return '1'
        else:
            return '0'


    def recognize(self, binary_imageORI):
        binary_image = binary_imageORI.copy()
        crop_bottom = binary_image[:(binary_image.shape[0]*3)//4, :]
        cut = crop_bottom.shape[1]//2
        r_image = crop_bottom[:, cut:-10]
        l_image = crop_bottom[:, 10:cut]
        # Find the columns of all black as a list
        l = []
        for i in range(l_image.shape[1]):
            if np.all(l_image[:, i] < 20):   # Check if the entire column is black
                l.append(i+10)
        r = []
        for i in range(r_image.shape[1]):
            if np.all(r_image[:, i] < 20):   # Check if the entire column is black
                r.append(i+cut)
        l_cut = self.find_longest_continuous_subarray_mid(l)
        r_cut = self.find_longest_continuous_subarray_mid(r)
        nums = [binary_image[:, :l_cut], binary_image[:, l_cut:r_cut], binary_image[:, r_cut:]]

        name = ['x', 'x', 'x']
        for n in range(3):
            num = nums[n]
            
            original = num.copy()
            num[original <= 128] = 0
            num[original > 128] = 255
            for i in range(len(num)):
                num[i] = self.find_longest_subarray_and_replace(num[i])

            contours, _ = cv2.findContours(num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a new binary image with the largest connected component
            new_binary_image = np.zeros_like(num)
            cv2.drawContours(new_binary_image, [largest_contour], 0, 255, -1)

            name[n] = self.detection(new_binary_image)
        
        if name[1] == '1':
            name[1] = '0'
            print('Error overcomed.')
        name = ''.join(name)
        
        return name


####### END OF HANDWRITING DIGIT RECOGNITION #######
####### Data saving functions and Main #######

    # csv that contains all the information extracted 
    def create_csv_all_data(self):
        # Sort the DataFrame by 'date' column
        self.df.sort_values(by='date', inplace=True)

        # Iterate over rows and update 'experiment_number' based on changing QRcode
        current_qrcode = None
        experiment_count = 0

        for index, row in self.df.iterrows():
            if current_qrcode is None or current_qrcode != row['QRcode']:
                current_qrcode = row['QRcode']
                experiment_count += 1
            self.df.at[index, 'experiment_number'] = experiment_count

        self.df.to_csv(ALL_DATA_CSV_PATH, index=False)
        print('All data extracted saved to', ALL_DATA_CSV_PATH)


    # csv that contains the spesified info in the config file
    def create_csv_data(self):
        columns_to_read = values
        df = pd.read_csv(ALL_DATA_CSV_PATH, usecols=columns_to_read, dtype=str)
        # Reorder the columns to match the order in the 'usecols' list
        df = df[columns_to_read]
        df.to_csv(DATA_CSV_PATH, index=False)
        print('Spesified data extracted saved to', DATA_CSV_PATH)


    def main(self):
        image_paths = self.find_image_paths(RAW_IMAGES_PATH)
        c = 0
        for image_path in image_paths:
            c += 1
            print(c, image_path)
            info, gray_cropped_image = self.rotate_name_crop(image_path)
            binary_image_digits = self.crop_gray_to_binary(gray_cropped_image)
            light = self.recognize(binary_image_digits)
            info = info.replace('xxx', light)
            data = info.split('-')
            # adding data to dataframe
            r = {'date':data[2], 'QRcode':data[0], 'experiment_number':0, 'light':data[1], 'phone_name':data[3], 'image_name':data[4]}
            self.df = pd.concat([self.df, pd.DataFrame([r])], ignore_index=True)

        self.create_csv_all_data()
        self.create_csv_data()


####### END OF Data saving functions and Main #######


Decoder = Decode_QR_Symbols()
