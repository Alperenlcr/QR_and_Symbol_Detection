# QR_and_Symbol_Detection

This repository has been created with the objective of extracting information from images used in experiment tracking. Its purpose is to read the information encoded in QR codes and identify handwritten symbols. The repository contains a total of 100 example images. The project serves as an automated data extraction tool for newly acquired experiment images. This work is a part of the process that helps the German laboratory and the Josef Stephan Institute in Slovenia communicate and work together smoothly.

### How to Run

- The python libraries take up some space so running it on virtual environment is recommended.
- Project developped with Python 3.8.10 so it is recommended to have no problem.
1. Clone the Repo
2. Create Virtual Environment
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
5. Edit the `config.py` as your own computer and according to your wishes
6. Run
    ```
    $ python3 main.py
    ```

## Code Explanation Step by Step

1. Finding image paths in the spesified path (RAW_IMAGES_PATH) in the config.py
2. Iterating on images to gathering aimed information.

    Example Images:

    ![IMG_20230725_100437](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/f7cc1f7a-bd0c-4463-9882-87b5c5ac6913)

    ![20230725_104732](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/34dc0f95-9e29-400d-ba39-d61f8fe0f1c5)

3. The images are not in same direction. If image is horizantal it rotates to vertical. In that case they can be normal or upside-down only.
4. Detection QR codes with the help of [QReader library](https://pypi.org/project/qreader/).
5. If the QR code is located in the downside than it is rotated 180°.
6. The handwriting symbols are located at the left of QR code. So the image is cropped.

    Example Images:

    ![289562-xxx-2023:07:25 10:52:45-Iphone 13pro-IMG_1959](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/bbac2c79-f999-4cb4-8fd3-edebcf810bb9)

    ![301948-xxx-2023:07:25 11:01:57-Samsung S22-20230725_110157](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/29b4fff9-9dce-464d-8723-fa11d4cac8aa)

7. The binarization operation is applied to cropped image as preprocessing for next steps. [Kraken library](https://pypi.org/project/kraken/) is used for that.

8. More cropping procedures are employed using a pixel similarity format and then manipulated accordingly.

    Example Images:

    ![289562-xxx-2023:07:25 10:51:59-Iphone 13pro-IMG_1958](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/96f90bc1-0105-45a9-805a-f04c863da529)

    ![289562-xxx-2023:07:25 10:52:37-Samsung S22-20230725_105237](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/a8f5a419-1ae8-4883-add5-b2dc4133cb36)

    ![774740-xxx-2023:07:25 11:11:32-Iphone 13pro-IMG_1981](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/fc9c6be2-a4d5-4304-b6d9-3625afc54ef2)

9. Further cropping and enhancing the image using specific operations for last preprocess step.

    Example Images:

    ![289562-xxx-2023:07:25 10:52:33-Huawei P60 pro-IMG_20230725_105233](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/e07303c6-b77d-44f7-914f-0e4a7df4dff5)

    ![301948-xxx-2023:07:25 10:59:22-Huawei P60 pro-IMG_20230725_105922](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/6d5ae016-602a-4715-bcf1-a61492857462)

    ![301948-xxx-2023:07:25 10:59:05-Honor Magic Vs-IMG_20230725_105905](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/a65ae55a-4a86-4efc-ba6a-59557ca8ef4d)

10. Recognizing algorithm:

    1. Divide into 3 subimages.
    2. Iterate through each row, identifying the longest sequence that begins and ends with white pixels, containing only black pixels.
    3. Compare each row with the previous one to determine whether its length has increased or decreased.
    4. Finally, if the ratio of (decreased/total)*100 is over 15%, the value is 0; otherwise, it's 1.

11. Finally, detecting the image's creation date and storing all the collected information.

    ![plant](https://github.com/Alperenlcr/QR_and_Symbol_Detection/assets/75525649/e83ba821-851b-4f4d-ab0a-77a45eef0f2c)
