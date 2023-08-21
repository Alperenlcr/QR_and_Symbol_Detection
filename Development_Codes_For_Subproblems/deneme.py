import cv2
import numpy as np

def perform_template_matching(image_path, template_path):
    # Load the main image and the template image
    main_image = cv2.imread(image_path)
    template_image = cv2.imread(template_path)

    # Convert both images to grayscale
    gray_main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    gray_template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(gray_main_image, gray_template_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Get the top-left and bottom-right corners of the matched area
    top_left = max_loc
    h, w = gray_template_image.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw a rectangle around the matched area on the main image
    matched_image = main_image.copy()
    cv2.rectangle(matched_image, top_left, bottom_right, (0, 255, 0), 2)

    # Display the matched image
    cv2.imshow('Matched Image', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the paths to the main image and template image
main_image_path = '/home/alperenlcr/Code/Plant_Grow_Tracking/main2.jpg'
template_image_path = '/home/alperenlcr/Code/Plant_Grow_Tracking/template.jpg'
# perform_template_matching(main_image_path, template_image_path)


def make_objects_bolder(image_path, thickness):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to create a mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw thicker contours on a blank canvas
    bolder_image = np.zeros_like(image)
    cv2.drawContours(bolder_image, contours, -1, (0, 255, 0), thickness)
    
    # Combine the original image and the thicker contours
    result = cv2.addWeighted(image, 1, bolder_image, 1, 0)
    
    # Save the modified image
    # Display the matched image
    cv2.imshow('Before', image)
    cv2.imshow('After', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
bolder_thickness = 5  # Adjust this thickness value as needed

#make_objects_bolder(main_image_path, bolder_thickness)


def enhance_image(image_path,  contrast_factor, sharpening_factor):
    # Load the image
    image = cv2.imread(image_path)
    
    # Apply contrast enhancement
    enhanced_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    
    # Apply sharpening using kernel
    kernel = np.array([[0, -sharpening_factor, 0],
                       [-sharpening_factor, 1 + 4 * sharpening_factor, -sharpening_factor],
                       [0, -sharpening_factor, 0]])
    
    sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)
    
    cv2.imshow('Before', image)
    cv2.imshow('After', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Example usage
contrast_factor = 1.5  # Adjust this contrast factor as needed
sharpening_factor = 0.5  # Adjust this sharpening factor as needed

#enhance_image(main_image_path, contrast_factor, sharpening_factor)



def draw_contours(image_path, output_path, contour_color=(0, 255, 0), contour_thickness=2):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find contours in the image
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on a blank canvas
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, contour_color, contour_thickness)
    
    # Combine the original image and the contour image
    result = cv2.addWeighted(image, 0.8, contour_image, 0.5, 0)
    
    # Save the modified image
    cv2.imshow('After', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
contour_color = (0, 0, 255)  # Red color
contour_thickness = 2

#draw_contours(main_image_path, contour_color, contour_thickness)


def sharpen_image(image_path, sharpening_factor=1.5):
    # Load the image
    image = cv2.imread(image_path)
    
    # Create a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Apply the sharpening kernel
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    # Save the sharpened image
    cv2.imshow('After', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage

sharpening_factor = 1.5  # Adjust this factor for stronger or weaker sharpening

#sharpen_image(main_image_path, sharpening_factor)



def edge_sharpening(image_path, alpha=2.0, beta=-1.0):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian operator to detect edges
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Normalize the Laplacian result to the range [0, 255]
    laplacian_normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Combine the original image with the edge-detected image
    sharpened_image = cv2.addWeighted(gray, alpha, laplacian_normalized, beta, 0)
    
    # Save the sharpened image
    cv2.imshow('Before', image)
    cv2.imshow('After', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
alpha = 2.0  # Adjust this factor for edge enhancement
beta = -1.0  # Negative value for unaltered original image

edge_sharpening(main_image_path, alpha, beta)


