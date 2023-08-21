# QR_and_Symbol_Detection

This repository has been created with the objective of extracting information from images used in experiment tracking. Its purpose is to read the information encoded in QR codes and identify handwritten symbols. The repository contains a total of 25 example images. The project serves as an automated data extraction tool for newly acquired experiment images. This work is a part of the process that helps the German laboratory and the Jozef Stefan Institute in Slovenia communicate and work together smoothly.

## How to Execute

- It is recommended to run the code within a virtual environment due to the space occupied by Python libraries.
- The project was developed using Python 3.8.10; compatibility issues are unlikely.
1. Clone the Repository
2. Create a Virtual Environment
    ```
    $ sudo apt-get install python3-venv
    $ cd <PathToRepo>/
    $ python3 -m venv .venv
    ```
3. Activate the Environment
    ```
    $ source .venv/bin/activate
    ```
4. Install Dependencies
    ```
    $ pip install -r requirements.txt
    ```
5. Configure the `config.py` and set the PATH variable.
6. Execute the Following Command
    ```
    $ python3 main.py
    ```

## Step-by-Step Code Explanation

1. Locate image paths within the specified directory (RAW_IMAGES_PATH) in `config.py`.
2. Iterate through images to gather the desired information.

    Example Images:

    ![Image 1](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/f7cc1f7a-bd0c-4463-9882-87b5c5ac6913)

    ![Image 2](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/34dc0f95-9e29-400d-ba39-d61f8fe0f1c5)

3. Adjust the orientation of images as needed; horizontal images are rotated vertically, resulting in normal or upside-down orientations.
4. Employ the [QReader library](https://pypi.org/project/qreader/) to detect QR codes.
5. Rotate the image by 180° if the QR code is found at the bottom.
6. Crop the image to the left of the QR code, where handwritten symbols are located.

    Example Images:

    ![Image 3](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/bbac2c79-f999-4cb4-8fd3-edebcf810bb9)

    ![Image 4](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/29b4fff9-9dce-464d-8723-fa11d4cac8aa)

7. Apply binarization to the cropped image for preprocessing using the [Kraken library](https://pypi.org/project/kraken/).
8. Perform additional cropping using a pixel similarity format and apply relevant manipulations.

    Example Images:

    ![Image 5](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/96f90bc1-0105-45a9-805a-f04c863da529)

    ![Image 6](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/a8f5a419-1ae8-4883-add5-b2dc4133cb36)

    ![Image 7](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/fc9c6be2-a4d5-4304-b6d9-3625afc54ef2)

9. Further refine the images through additional cropping and enhancement using specified operations.

    Example Images:

    ![Image 8](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/e07303c6-b77d-44f7-914f-0e4a7df4dff5)

    ![Image 9](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/6d5ae016-602a-4715-bcf1-a61492857462)

    ![Image 10](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/a65ae55a-4a86-4efc-ba6a-59557ca8ef4d)

10. Recognition Algorithm:

    1. Divide the image into 3 subimages.
    2. Iterate through each row, identifying the longest sequence of black pixels starting and ending with white pixels.
    3. Compare each row with the previous one to determine whether its length has increased or decreased.
    4. If the ratio of (decreased/total)*100 is above 15%, the value is 0; otherwise, it's 1.

11. Conclude by detecting the image's creation date and preserving all the gathered information.

    ![Plant](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/e83ba821-851b-4f4d-ab0a-77a45eef0f2c)
