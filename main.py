from config import *


class Analysis():
    def __init__(self):
        os.makedirs(SAVED_IMGS_PATH, exist_ok=True)


    # crops image from given rectangle coordinates and makes padding
    def crop_rectangle_and_padding(self, image, corners, thickness):
        # draw in red
        cv2.line(image, corners['lt'], corners['rt'], (0, 0, 255), thickness)  # Blue line from left-top to right-top
        cv2.line(image, corners['rt'], corners['rb'], (0, 0, 255), thickness)  # Green line from right-top to right-bottom
        cv2.line(image, corners['rb'], corners['lb'], (0, 0, 255), thickness)  # Red line from right-bottom to left-bottom
        cv2.line(image, corners['lb'], corners['lt'], (0, 0, 255), thickness)  # Cyan line from left-bottom to left-top

        # crop containing rectangle
        containing_rectangle_lt = [corners['lt'][0] if corners['lt'][0] < corners['lb'][0] else corners['lb'][0],
                                    corners['lt'][1] if corners['lt'][1] < corners['rt'][1] else corners['rt'][1]]
        containing_rectangle_rb = [corners['rt'][0] if corners['rt'][0] > corners['rb'][0] else corners['rb'][0],
                                   corners['lb'][1] if corners['lb'][1] > corners['rb'][1] else corners['rb'][1]]
        containing_rectangle_img = image[containing_rectangle_lt[1]:containing_rectangle_rb[1], containing_rectangle_lt[0]:containing_rectangle_rb[0]]

        # padding:
        height, width, _ = containing_rectangle_img.shape
        for row_index in range(height):
            cloumn_index = 0
            while cloumn_index<width and not np.array_equal(containing_rectangle_img[row_index, cloumn_index], [0, 0, 255]):
                containing_rectangle_img[row_index][cloumn_index] = [0, 0, 0]
                cloumn_index += 1
            l = cloumn_index

            cloumn_index = width-1
            while cloumn_index>0 and not np.array_equal(containing_rectangle_img[row_index, cloumn_index], [0, 0, 255]):
                containing_rectangle_img[row_index][cloumn_index] = [0, 0, 0]
                cloumn_index -= 1
            r = cloumn_index

            cloumn_index = l
            while cloumn_index<r and np.array_equal(containing_rectangle_img[row_index, cloumn_index], [0, 0, 255]):
                containing_rectangle_img[row_index][cloumn_index] = [0, 0, 0]
                cloumn_index += 1

            cloumn_index = r
            while cloumn_index>l and np.array_equal(containing_rectangle_img[row_index, cloumn_index], [0, 0, 255]):
                containing_rectangle_img[row_index][cloumn_index] = [0, 0, 0]
                cloumn_index -= 1


        return containing_rectangle_img


    # removes all the rows and columns that has only black pixels
    def remove_black_rows_columns(self, image, specialized=100):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find rows and columns with all black pixels (0 in grayscale)
        black_rows = np.all(gray <= specialized, axis=1)
        black_columns = np.all(gray <= specialized, axis=0)

        # Remove black rows and columns
        result = image[~black_rows, :]
        result = result[:, ~black_columns]

        return result


    # calculates distance to a object with a known size
    def calculate_distance_to_object_similarity(self, black_rectangle_corners, image_path, rotated_bgr_image):
        width, height, horizontal_resolution, vertical_resolution, focal_length, focal_length_35mm = self.read_metadata(image_path, rotated_bgr_image)

        W_pixels_t = sqrt((black_rectangle_corners['lt'][0] - black_rectangle_corners['rt'][0])**2 + (black_rectangle_corners['lt'][1] - black_rectangle_corners['rt'][1])**2)
        W_pixels_b = sqrt((black_rectangle_corners['lb'][0] - black_rectangle_corners['rb'][0])**2 + (black_rectangle_corners['lb'][1] - black_rectangle_corners['rb'][1])**2)

        focal_length = round(focal_length, 1)
        fp = 3765 # for focal length 5.5 / 3475 for focal length 5.5
        if focal_length < 6:
            fp += (focal_length - 5.5) * 800
        else:
            fp += 320 + (focal_length - 5.9) * 100

        dt = (109.4 * fp) / W_pixels_t
        db = (109.4 * fp) / W_pixels_b
        return db, dt


    # reads image metadata
    def read_metadata(self, image_path, image):
        height, width = image.shape[:2]
        horizontal_resolution, vertical_resolution, focal_length, focal_length_35mm = 0, 0, 0, 0
        
        # Open the image
        image = IMG.open(image_path)
        # Extract Exif data
        exif_data = image._getexif()

        # Search for the focal length tag (some common tag IDs)
        for tag, value in exif_data.items():
            if TAGS.get(tag) == 'XResolution':
                horizontal_resolution += float(value)
            elif TAGS.get(tag) == 'YResolution':
                vertical_resolution += float(value)
            elif TAGS.get(tag) == 'FocalLength':
                focal_length += float(value)
            elif TAGS.get(tag) == 'FocalLengthIn35mmFilm':
                focal_length_35mm += float(value)

        if focal_length_35mm == 0 and 'Redmi' in image_path:    # It is not written in metadata
            focal_length_35mm = 25.9    # https://www.camerafv5.com/devices/manufacturers/xiaomi/2201116tg_viva_0/

        return width, height, horizontal_resolution, vertical_resolution, focal_length, focal_length_35mm


    # detects the biggest contour in image and returns the coordinates of rectangle
    def detect_black_rectangle(self, rotated_bgr_image, c, draw_rectangle=False):
        original = rotated_bgr_image
        height = rotated_bgr_image.shape[0]
        width = rotated_bgr_image.shape[1]
        rotated_bgr_image = cv2.resize(rotated_bgr_image, (width//5, height//5))
        # Convert the OpenCV image (NumPy array) to a Pillow image
        pil_image = IMG.fromarray(rotated_bgr_image[:,:-8])
        # Apply binarization using Kraken's binarization module
        binary_pil_image = binarization.nlbin(pil_image)  # Simple Niblack binarization
        # Convert the Pillow binary image back to a NumPy array for displaying
        binary_image = np.array(binary_pil_image)
        binary_image = cv2.bitwise_not(binary_image)
        # Find contours
        contours, hierarchy = cv2.findContours(image=binary_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # Find the index of the biggest contour based on the contour area
        biggest_contour_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        liste = [x[0] for x in contours[biggest_contour_idx].tolist()]
        tree = spatial.KDTree(liste)

        p1 = liste[tree.query([(0, 0)])[1][0]]
        p1 = [p1[0]*5, p1[1]*5]

        p2 = liste[tree.query([(width//5, 0)])[1][0]]
        p2 = [p2[0]*5, p2[1]*5]

        p3 = liste[tree.query([(0, height//5)])[1][0]]
        p3 = [p3[0]*5, p3[1]*5]

        p4 = liste[tree.query([(width//5, height//5)])[1][0]]
        p4 = [p4[0]*5, p4[1]*5]

        if draw_rectangle == True:
            # left top
            cv2.line(original, (p1[0], 0), p1, (0, 0, 255), 4)
            cv2.putText(original, str(p1[0]), (p1[0]//2, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.line(original, (0, p1[1]), p1, (0, 0, 255), 4)
            cv2.putText(original, str(p1[1]), (p1[0], p1[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            # right top
            cv2.line(original, p2, (width, p2[1]), (0, 0, 255), 4)
            cv2.putText(original, str(width - p2[0]), (p2[0]+(width - p2[0])//2, p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.line(original, (p2[0], 0), p2, (0, 0, 255), 4)
            cv2.putText(original, str(p2[1]), (p2[0], p1[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            # left bottom
            cv2.line(original, (0, p3[1]), p3, (0, 0, 255), 4)
            cv2.putText(original, str(p3[0]), (p3[0]//2, p3[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.line(original, (p3[0], height), p3, (0, 0, 255), 4)
            cv2.putText(original, str(height - p3[1]), (p3[0], p3[1]+(height - p3[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            # right bottom
            cv2.line(original, p4, (width, p4[1]), (0, 0, 255), 4)
            cv2.putText(original, str(width - p4[0]), (p4[0]+(width - p4[0])//2, p4[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.line(original, (p4[0], height), p4, (0, 0, 255), 4)
            cv2.putText(original, str(height - p4[1]), (p4[0], p4[1]+(height - p4[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            cv2.imwrite(str(c)+".jpg", original)
        return {'lt':p1, 'rt':p2, 'lb':p3, 'rb':p4}, original


    # detects the 12 circles which are inner and outer and returns the info
    def detect_circles(self, img):
        def sorts(data):
            # Sort based on the second element of each sublist
            sorted_data = list(sorted(data, key=lambda x: int(x[0])))

            # Split the sorted data into two halves
            mid = len(sorted_data) // 2
            first_half = sorted_data[:mid]
            second_half = sorted_data[mid:]

            # Sort each half based on the third element of each sublist
            first_half_sorted = list(sorted(first_half, key=lambda x: int(x[1])))
            second_half_sorted = list(sorted(second_half, key=lambda x: int(x[1])))

            # Combine the sorted halves back into a single list
            return first_half_sorted + second_half_sorted

        img = img.copy()
        # Convert to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur using 5 * 5 kernel.
        gray_blurred = cv2.blur(gray, (5, 5))
# BIG
        big_circle_centers = []
        height, width = img.shape[:2]
        R = int((width - width/8.3)/4)
        detected_circles_big = [[0]]
        threshold = 55
        f = False
        while threshold > 15 and len(detected_circles_big[0]) != 6:
            f = False
            # Apply Hough transform on the blurred image.
            detected_circles_big = cv2.HoughCircles(gray_blurred, 
                            cv2.HOUGH_GRADIENT, 1, int(1.5*R), param1 = 50,
                        param2 = threshold, minRadius = R-width//50, maxRadius = R+width//50)
            threshold -= 1
            if type(detected_circles_big) == type(None):
                detected_circles_big = [[0]]
                f = True
        if f:
            detected_circles_big = None

        # Draw circles that are detected.
        if detected_circles_big is not None:
            # Convert the circle parameters a, b and r to integers.
            detected_circles_big = np.int32(np.around(detected_circles_big))
        
            for pt in detected_circles_big[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
        
                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (255, 255, 0), 4)
        
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (255, 255, 0), 12)
                big_circle_centers.append([a, b, r])
# SMALL
        small_circle_centers = []
        r = R - 80
        detected_circles_small = [[0]]
        threshold = 55
        f = False
        while threshold > 15 and len(detected_circles_small[0]) != 6:
            f = False
            # # Apply Hough transform on the blurred image.
            detected_circles_small = cv2.HoughCircles(gray_blurred, 
                            cv2.HOUGH_GRADIENT, 1, int(1.25*r), param1 = 50,
                        param2 = threshold, minRadius = r-width//75, maxRadius = r+width//60)
            threshold -= 1
            if type(detected_circles_small) == type(None):
                detected_circles_small = [[0]]
                f = True
        if f:
            detected_circles_small = None

        # Draw circles that are detected.
        if detected_circles_small is not None:
            # Convert the circle parameters a, b and r to integers.
            detected_circles_small = np.int32(np.around(detected_circles_small))

            for pt in detected_circles_small[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
        
                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (255, 0, 255), 4)
        
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (255, 0, 255), 12)
                small_circle_centers.append([a, b, r])
        
        return img, sorts(big_circle_centers), sorts(small_circle_centers)


    # after preprocess applies warp perspective
    def normalize(self, image, shape):
        # Warp Image
        # Remove black rows and columns from the image
        image = self.remove_black_rows_columns(image)

        # Threshold the image
        threshold_value = 100
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholded_image = np.where(gray_image < threshold_value, 0, gray_image)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(image=thresholded_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # Find the index of the biggest contour based on contour area
        biggest_contour_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

        # Get the dimensions of the original image
        height, width = image.shape[:2]

        # Get the minimum area rotated rectangle for the biggest contour
        rotated_rect = cv2.minAreaRect(contours[biggest_contour_idx])

        # Convert the rotated rectangle to a set of four corner points
        box = cv2.boxPoints(rotated_rect)
        box = np.float32(box)
        box = np.clip(box, a_min=0, a_max=height-1)

        corners = [[],[],[],[]]
        for p in box:
            if p[0] < width/2:
                if p[1] < height/2:
                    corners[0] = p
                else:
                    corners[2] = p
            else:
                if p[0] > width-1:
                    p[0] = width-1
                if p[1] < height/2:
                    corners[1] = p
                else:
                    corners[3] = p

        # Define the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [width, 0], [0, height], [width, height]]))

        # Warp the original image to crop the rotated rectangle
        image = cv2.warpPerspective(image, matrix, (width, height))

        # Cut Yellow
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        max_pixel_count = 0
        cut_index = None
        # Iterate through rows in the grayscale image / Count the number of pixels in the specified range (160-210)
        for row_index, row in enumerate(gray[:int(gray.shape[1]*0.1)]):
            pixel_count_in_range = np.sum((row >= 160) & (row <= 210))
            if pixel_count_in_range > max_pixel_count:
                max_pixel_count = pixel_count_in_range
                cut_index = row_index

        image = image[cut_index:]

        # Warp Image
        # Apply Canny edge detection
        edges = cv2.Canny(image, threshold1=50, threshold2=150, apertureSize=3)
        # Find the indices of white pixels (non-zero values) using numpy.where()
        white_pixel_indices = np.where(edges == 255)
        height, width = image.shape[:2]

        # Create an array of (x, y) coordinates for white pixels
        white_pixel_coordinates = np.column_stack((white_pixel_indices[1], white_pixel_indices[0]))

        tree = spatial.KDTree(white_pixel_coordinates)

        p1 = white_pixel_coordinates[tree.query([(0, 0)])[1][0]]
        p1 = [p1[0], p1[1]]

        p2 = white_pixel_coordinates[tree.query([(width, 0)])[1][0]]
        p2 = [p2[0], p2[1]]

        p3 = white_pixel_coordinates[tree.query([(0, height)])[1][0]]
        p3 = [p3[0], p3[1]]

        p4 = white_pixel_coordinates[tree.query([(width, height)])[1][0]]
        p4 = [p4[0], p4[1]]

        desired_width, desired_height = width, int(width*shape)
        points_trapezoid = np.float32([p1, p2, p3, p4])
        points_rectangle = np.float32([[0, 0], [desired_width, 0], [0, desired_height], [desired_width, desired_height]])
        matrix = cv2.getPerspectiveTransform(points_trapezoid, points_rectangle)
        image = cv2.warpPerspective(image, matrix, (desired_width, desired_height))

        return image



    # checks the circle positions to determine anormallies
    def check_anormally(self, bcc, scc):
    # check if bcc and scc looks good
        bcc = np.array([sublist[:2] for sublist in bcc], dtype=np.int32)
        scc = np.array([sublist[:2] for sublist in scc], dtype=np.int32)

        # Create an empty list to store points outside the bcc contour
        points_outside_bcc = []

        # Iterate through each point in scc and check if it's outside the bcc contour
        bcc_polygon = np.array([bcc[2], bcc[1], bcc[0], bcc[3], bcc[4], bcc[5]], dtype=np.float32)
        for point in scc:
            distance = cv2.pointPolygonTest(bcc_polygon, (float(point[0]), float(point[1])), True)
            if distance < -8:
                points_outside_bcc.append(point)

        count = len(points_outside_bcc)
        if count > 1:
            return 'Wrong Circle Detection'

        # forecast the outlier if there is
        if count == 1:
            index = np.where(scc == points_outside_bcc[0])[0][0]
            vps = []
            if index in [0,1,2]:
                for i in [0,1,2]:
                    if index != i:
                        vps.append(scc[i][0])
            if index in [3,4,5]:
                for i in [3,4,5]:
                    if index != i:
                        vps.append(scc[i][0])
            new_p = [(vps[0]+vps[1])//2, scc[(index+3)%6][1]]
            scc[index] = new_p

        # Splitting bcc and scc into vertical components
        bcc1v, bcc2v = bcc[:3], bcc[3:]
        scc1v, scc2v = scc[:3], scc[3:]

        # Splitting splitting bcc and scc horizontal components
        bcc1h, bcc2h, bcc3h = abs(bcc[0][1]-bcc[3][1]), abs(bcc[1][1]-bcc[4][1]), abs(bcc[2][1]-bcc[5][1])
        scc1h, scc2h, scc3h = abs(scc[0][1]-scc[3][1]), abs(scc[1][1]-scc[4][1]), abs(scc[2][1]-scc[5][1])

        x1 = sorted([pair[0] for pair in bcc1v])
        x2 = sorted([pair[0] for pair in bcc2v])
        bdistance1 = abs(x1[2]-x1[0])
        bdistance2 = abs(x2[2]-x2[0])

        x1 = sorted([pair[0] for pair in scc1v])
        x2 = sorted([pair[0] for pair in scc2v])
        sdistance1 = abs(x1[2]-x1[0])
        sdistance2 = abs(x2[2]-x2[0])

        if max([bdistance1, bdistance2, sdistance1, sdistance2, bcc1h, bcc2h, bcc3h, scc1h, scc2h, scc3h]) > 50:
            return 'Wrong Circle Detection'

        # check if scc is correct according to bcc
        sumh, sumv = 0, 0
        for i in range(6):
            sumh += bcc[i][0]-scc[i][0]
            sumv += bcc[i][1]-scc[i][1]

        return f'Normal, x angle multiplier:{sumv/20:.1f}, y angle multiplier:{-sumh/10:.1f}'


#####################################################
#####################################################


class Decode_QR_Symbols_and_Analysis():
    def __init__(self):
        # Create a QReader instance
        self.A = Analysis()
        self.qreader = QReader()
        self.df = pd.DataFrame(columns=column_names)
        self.main()

#####################################################
####################### UTILS #######################


    # displays image
    def display(self, img, title='Image'):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Finds all images (including the ones in subfoler) in the given folder, returns their paths as list
    def find_image_paths(self, path):
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


#################### END OF UTILS ###################
#####################################################

#####################################################
###################### READ QR ######################


    # Get the image that contains the QR code (QReader expects an uint8 numpy array)
    def read_QR(self, bgr_image, original):
        # Method 1
        qrCodeDetector = cv2.QRCodeDetector()
        decodedText, points, _ = qrCodeDetector.detectAndDecode(bgr_image)
        QR_code = decodedText
        if QR_code in QR_codes:
            return QR_code

        # Method 2
        decoded_objects = decode(bgr_image, symbols=[ZBarSymbol.QRCODE])
        if len(decoded_objects) == 1:
            QR_code = decoded_objects[0].data.decode('utf-8')
        if QR_code in QR_codes:
            return QR_code

        # Method 3
        decoded_text = self.qreader.detect_and_decode(image=bgr_image, return_detections=True, is_bgr=True)
        QR_code = decoded_text[0][0]
        if len(decoded_text[0]) != 1:
            return None
        if QR_code in QR_codes:
            return QR_code

        # Method 4
        decoded_text = self.qreader.detect_and_decode(image=original, return_detections=True, is_bgr=True)
        QR_code = decoded_text[0][0]
        if len(decoded_text[0]) != 1:
            return None
        if QR_code in QR_codes:
            return QR_code
        return None


    def rotate(self, image_path):
        reader = BarcodeReader()
        try:
            text_results = reader.decode_file(image_path)
            text_result = text_results[0]
        except BarcodeReaderError as bre:
            print(bre)

        bgr_image = cv2.imread(image_path)
        original = bgr_image.copy()
        # Extract the coordinates of the rectangle's corners
        y_coordinates, x_coordinates = zip(*text_result.localization_result.localization_points)

        # Find the minimum and maximum X and Y coordinates to determine the bounding box
        min_x, min_y = min(x_coordinates), min(y_coordinates)
        max_x, max_y = max(x_coordinates), max(y_coordinates)

        # exception
        if 'Iphone' in image_path or 'Samsung' in image_path:
            tmax_y, tmin_y = max_y, min_y
            max_y, min_y = bgr_image.shape[1] - min_x, bgr_image.shape[1] - max_x
            max_x, min_x = tmax_y, tmin_y

        cy, cx = (min_x+max_x)//2, (min_y+max_y)//2
        # Crop the image using the bounding box
        gap = 100
        if gap > min(min_x, min_y):
            gap = min(min_x, min_y) - 1
        cropped_QR = bgr_image[min_x-gap:max_x+gap, min_y-gap:max_y+gap]
        QR_code = self.read_QR(cropped_QR, original)
        mid_x = bgr_image.shape[1] // 2
        mid_y = bgr_image.shape[0] // 2

        if cx < mid_x:
            if cy < mid_y:
                min_x, max_x = min_y, max_y
                bgr_image = cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE)  # Left Top Quarter
            else:
                min_x, max_x = bgr_image.shape[0] - max_x, bgr_image.shape[0] - min_x
                bgr_image = cv2.rotate(bgr_image, cv2.ROTATE_180)   # Left Bottom Quarter
        else:
            if cy < mid_y:
                pass    # Right Top Quarter
            else:
                min_x, max_x = bgr_image.shape[1] - max_y, bgr_image.shape[1] - min_y
                bgr_image = cv2.rotate(bgr_image, cv2.ROTATE_90_COUNTERCLOCKWISE)   # Right Bottom Quarter

        return bgr_image, QR_code, min_x, max_x


    def info(self, image_path, QR_code):
        image_name = image_path.split('/')[-1].split('.')[0]
        phone_name = image_path.split('/')[-2]
        date = self.get_original_capture_date(image_path)
        light = 'xxx'
        file_name = f'{QR_code}-{light}-{date}-{phone_name}-{image_name}.jpg'
        return file_name


    def name_crop(self, image_path, rotated_bgr_image, QR_code, min_x, black_rectangle_corners):
        W_pixels_t = sqrt((black_rectangle_corners['lt'][0] - black_rectangle_corners['rt'][0])**2 + (black_rectangle_corners['lt'][1] - black_rectangle_corners['rt'][1])**2)
        max_x = int(min(black_rectangle_corners['lt'][1], black_rectangle_corners['rt'][1]))
        min_y = int(black_rectangle_corners['lt'][0])
        max_y = int(min_y + W_pixels_t // 4)
        min_x -= 20
        info = self.info(image_path, QR_code)
        if QR_code in QR_codes:
            gray_image = cv2.cvtColor(rotated_bgr_image, cv2.COLOR_BGR2GRAY)
            return info, gray_image[min_x:max_x, min_y:max_y]
        else:
            return info, cv2.imread(image_path)


################## END OF READ QR ###################
#####################################################

#####################################################
########### HANDWRITING DIGIT RECOGNITION ###########


    def crop_gray_to_binary(self, image):
        # RIGHT
        binary_image = cv2.Canny(image, threshold1=50, threshold2=150, apertureSize=3)
        original = binary_image.copy()
        binary_image[original <= 128] = 255
        binary_image[original > 128] = 0
        mid = binary_image[binary_image.shape[0]//6:(binary_image.shape[0]*5)//6, :]
        right_black = 0    # Find the rightmost blackness
        bottom_index = 0
        bold = 5
        for i in range(mid.shape[1]):
            column = mid[:, i]
            if not np.all(column > 230):  # Check if the entire column is not white
                bold -= 1
                if bold <= 0:
                    right_black = i
                    bottom_index = np.where(column == 0)[0][-1] + binary_image.shape[0]//6
            else:
                bold = 5
        gap = 100
        if gap >= bottom_index:
            gap = bottom_index - 1
        image = image[bottom_index-gap:bottom_index+gap, :right_black]
        
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

            f = True
            while f or len(contours) == 0:
                if f == False:
                    print()
                f = False
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


####### END OF HANDWRITING DIGIT RECOGNITION ########
#####################################################

#####################################################
########### Data saving functions and Main ##########


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


    # all processes on an image 
    def process(self, temp):
        c, image_path = temp
        print(c, image_path)
        rotated_bgr_image, QR_code, min_x, max_x = self.rotate(image_path)
        black_rectangle_corners, rotated = self.A.detect_black_rectangle(rotated_bgr_image.copy(), c, draw_rectangle=False) # from Analysis
        info, gray_cropped_image = self.name_crop(image_path, rotated_bgr_image.copy(), QR_code, min_x, black_rectangle_corners)
        binary_image_digits = self.crop_gray_to_binary(gray_cropped_image)
        light = self.recognize(binary_image_digits)
        info = info.replace('xxx', light)
        data = info.split('-')

        # Analysis
        black_rectangle_image = self.A.crop_rectangle_and_padding(rotated, black_rectangle_corners, thickness=100)
        plants_img = self.A.normalize(black_rectangle_image, 1.4065) # 128*0.945/86 -> value comes from https://www.tpp.ch/page/produkte/09_zellkultur_testplatte.php
        circle_detected, big_circle_centers, small_circle_centers = self.A.detect_circles(plants_img)
        status = self.A.check_anormally(big_circle_centers, small_circle_centers)
        height_phone_b, height_phone_t = self.A.calculate_distance_to_object_similarity(black_rectangle_corners, image_path, rotated_bgr_image)
        
        # Adding data to dataframe
        r = {'date':data[2], 'QRcode':data[0], 'experiment_number':0, 'light':data[1], 'phone_name':data[3], 'image_name':data[4], \
                'black_rectangle_corners':black_rectangle_corners, 'height_phone_b':round(height_phone_b, 6), 'height_phone_t':round(height_phone_t, 6), 'height_phone':round((height_phone_b+height_phone_t)/2, 6), 'circles_info':{'big_circle_centers':big_circle_centers, 'small_circle_centers':small_circle_centers}, 'angle_multiplier_info':status, 'savingID':c}
        self.df = pd.concat([self.df, pd.DataFrame([r])], ignore_index=True)
        # saving photos
        if save_normalized_tubes_with_circles:
            file_name = SAVED_IMGS_PATH+'C'+str(c)+'.jpg'
            cv2.imwrite(file_name, circle_detected)
        if save_normalized_tubes:
            file_name = SAVED_IMGS_PATH+'N'+str(c)+'.jpg'
            cv2.imwrite(file_name, plants_img)


    def main(self):
        image_paths = [x.replace('\\', '/') for x in self.find_image_paths(RAW_IMAGES_PATH)]

        # Paralel Runtime
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.process, list(enumerate(image_paths, start=1)))

        # Linear Runtime
        # c = 0
        # for image_path in image_paths:
        #     c += 1
        #     self.process((c, image_path))

        self.create_csv_all_data()
        self.create_csv_data()


####### END OF Data saving functions and Main #######
#####################################################


Decoder = Decode_QR_Symbols_and_Analysis()
