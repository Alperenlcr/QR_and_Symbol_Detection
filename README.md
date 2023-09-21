# QR_and_Symbol_Detection + Image Analysis

This repository has been created with the objective of extracting information from images used in experiment tracking. Its purpose is to read the information encoded in QR codes, identify handwritten symbols, detecting tube circles and some important coordinates, finding the distance to tubes and estimating the angle of the phone . The repository contains a total of 25 example images. The project serves as an automated data extraction tool for newly acquired experiment images. This work is helps the project which runs between a laboratory of Faculty of Chemistry and Chemical Technology and Jozef Stefan Institute in Slovenia.

# How to Execute

- It is recommended to run the code within a virtual environment due to the space occupied by Python libraries or in docker container.
- The project was developed using Python 3.8.10; compatibility issues are unlikely.

## How to Run in Virtual Environment

1. Clone the Repository
2. Create a Virtual Environment
    ```
    $ cd <PathToRepo>/QR_and_Symbol_Detection-Image_Analysis/
    $ python -m venv .venv
    ```
3. Activate the Environment
    ```
    $ source .venv/bin/activate
    ```
4. Install Dependencies
    ```
    $ pip install -r requirements.txt
    ```
5. Configure the **config.py** and set the `<PATH>` variables. Change the boolean values of `save_normalized_tubes` and `save_normalized_tubes_with_circles` to save images or not.
6. Execute the Following Command
    ```
    $ python main.py
    ```
## How to Run in Docker Container

- Install [docker](https://docs.docker.com/engine/install/) to your system.
- Change paths in `config.py` and `compose.yaml`.
    - Delete '\<PATH>' in `config.py`. It should appear as follows.
        ```
            RAW_IMAGES_PATH = 'Phones/'
            ALL_DATA_CSV_PATH = 'all_data.csv'
            DATA_CSV_PATH = 'data.csv'
            SAVED_IMGS_PATH = 'Saved_imgs/'
        ```
        You also have the flexibility to modify the phones folder name or CSV file names, as long as they remain adjacent to the main.py file.
- Modify this line in `compose.yaml`.
    ```
        - <YOUR_PATH>/QR_and_Symbol_Detection-Image_Analysis:/app  # This is necessary for saving output
    ```
- Run this command to create and run container.
    ```
        docker compose up --build
    ```

# Step-by-Step Code Explanation
## QR and Symbol_Detection Part
1. Locate image paths within the specified directory (RAW_IMAGES_PATH) in `config.py`.
2. Iterate through images to gather the desired information.

    Example Images:

    ![Image 1](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/f7cc1f7a-bd0c-4463-9882-87b5c5ac6913)

    ![Image 2](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/34dc0f95-9e29-400d-ba39-d61f8fe0f1c5)

3. Adjust the orientation of images as needed; horizontal images are rotated vertically, resulting in normal or upside-down orientations.
4. Employ the [QReader library](https://pypi.org/project/qreader/) to detect QR codes.
5. Rotate the image by 180° if the QR code is found at the bottom.
6. Crop the image to the left of the QR code, where handwritten symbols are located.

    Example Images:

    ![Image 3](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/bbac2c79-f999-4cb4-8fd3-edebcf810bb9)

    ![Image 4](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/29b4fff9-9dce-464d-8723-fa11d4cac8aa)

7. Apply binarization to the cropped image for preprocessing using the [Kraken library](https://pypi.org/project/kraken/).
8. Perform additional cropping using a pixel similarity format and apply relevant manipulations.

    Example Images:

    ![Image 5](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/96f90bc1-0105-45a9-805a-f04c863da529)

    ![Image 6](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/a8f5a419-1ae8-4883-add5-b2dc4133cb36)

    ![Image 7](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/fc9c6be2-a4d5-4304-b6d9-3625afc54ef2)

9. Further refine the images through additional cropping and enhancement using specified operations.

    Example Images:

    ![Image 8](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/e07303c6-b77d-44f7-914f-0e4a7df4dff5)

    ![Image 9](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/6d5ae016-602a-4715-bcf1-a61492857462)

    ![Image 10](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/a65ae55a-4a86-4efc-ba6a-59557ca8ef4d)

10. Recognition Algorithm:

    1. Divide the image into 3 subimages.
    2. Iterate through each row, identifying the longest sequence of black pixels starting and ending with white pixels.
    3. Compare each row with the previous one to determine whether its length has increased or decreased.
    4. If the ratio of (decreased/total)*100 is above 15%, the value is 0; otherwise, it's 1.

11. Conclude by detecting the image's creation date and preserving all the gathered information.

    ![Plant](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/e83ba821-851b-4f4d-ab0a-77a45eef0f2c)


## Image Analysis Part

1. Identify the black rectangle within images that have been rotated to their standard orientation.
2. Crop and apply padding to this identified rectangle.
3. Standardize the dimensions of the rectangular region containing the tubes.
4. Detect the circles of 12 circular tops and bottoms of the tubes.
![F9](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/2e2e542a-0e87-4d54-a660-65b9fdef3910)
5. Analyze the location data of these circles to determine the angle of the phone.
6. Compute the distance to the tubes, leveraging our knowledge of the tubes' actual dimensions.
7. Save images based on the configuration variables `save_normalized_tubes` and `save_normalized_tubes_with_circles` from the **config.py** file.

### Result Table Columns:
**Column names and an example row from `all_data.csv`**

| Attribute               | Value                   |
|-------------------------|-------------------------|
| Date                    |: 2023-07-25 10:00:59     |
| QR Code                 |: 705548                  |
| Experiment Number       |: 1                       |
| Light                   |: 101                     |
| Phone Name              |: Huawei P60 Pro          |
| Image Name              |: IMG_20230725_100059.jpg |
| Black Rectangle Corners |: {'lt': [610, 720], 'rt': [2380, 725], 'lb': [540, 3605], 'rb': [2500, 3585]} |
| Height (Bottom)         |: 230.230361              |
| Height (Top)            |: 254.95661               |
| Height                  |: 242.593485              |
| Circles Info            |: {'big_circle_centers': [[424, 392, 354], [410, 1078, 324], [422, 1786, 335], [1158, 366, 329], [1164, 1084, 316], [1150, 1788, 332]], 'small_circle_centers': [[440, 448, 291], [448, 1098, 291], [438, 1774, 285], [1150, 432, 267], [1136, 1108, 279], [1122, 1778, 287]]} |
| Angle Multiplier Info   |: Normal, x angle multiplier:-7.2, y angle multiplier:0.6 |
| Saving ID               |: 73                      |

![Screenshot 2023-09-19 143150](https://github.com/Alperenlcr/QR_and_Symbol_Detection-Image_Analysis/assets/75525649/6d04704f-8f9e-491d-8762-395bd041e1dd)
