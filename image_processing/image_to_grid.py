#!/usr/bin/env python3
import cv2
import numpy as np
from scipy import ndimage
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K


def init_sheet():
    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            row.append(0)
        grid.append(row)
    
    return grid


def image_to_grid(warpped_image):
    height = warp.shape[0] // 9
    width = warp.shape[1] // 9

    offset_width = math.floor(width / 10)    # Offset is used to get rid of the boundaries
    offset_height = math.floor(height / 10)
    
    grid = init_sheet()
    model = load_model()
    # Divide the Sudoku board into 9x9 square:
    for i in range(9):
        for j in range(9):

            # Crop with offset (We don't want to include the boundaries)
            crop_image = warp[(height * i + offset_height):(height * (i + 1) - offset_height), 
                              (width * j + offset_width):(width * (j + 1) - offset_width)]        
            
            # There are still some boundary lines left though
            # => Remove all black lines near the edges
            # ratio = 0.6 => If 60% pixels are black, remove
            # Notice as soon as we reach a line which is not a black line, the while loop stops
            ratio = 0.6        
            # Top
            while np.sum(crop_image[0]) <= (1 - ratio) * crop_image.shape[1] * 255:
                crop_image = crop_image[1:]
            # Bottom
            while np.sum(crop_image[:, -1]) <= (1 - ratio) * crop_image.shape[1] * 255:
                crop_image = np.delete(crop_image, -1, 1)
            # Left
            while np.sum(crop_image[:, 0]) <= (1-ratio) * crop_image.shape[0] * 255:
                crop_image = np.delete(crop_image, 0, 1)
            # Right
            while np.sum(crop_image[-1]) <= (1-ratio) * crop_image.shape[0] * 255:
                crop_image = crop_image[:-1]    

            # Take the largestConnectedComponent (The digit), and remove all noises
            crop_image = cv2.bitwise_not(crop_image)
            crop_image = largest_connected_component(crop_image)
        
            # Resize
            crop_image = cv2.resize(crop_image, (28, 28))


            # If this is a white cell, set grid[i][j] to 0 and continue on the next image:

            # Criteria 1 for detecting white cell:
            # Has too little black pixels
            digit_pic_size = 28
            if crop_image.sum() >= digit_pic_size**2 * 255 - digit_pic_size * 1 * 255:
                grid[i][j] == 0
                continue    # Move on if we have a white cell

            # Criteria 2 for detecting white cell
            # Huge white area in the center
            center_width = crop_image.shape[1] // 2
            center_height = crop_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = crop_image[x_start:x_end, y_start:y_end]
            
            if center_region.sum() >= center_width * center_height * 255 - 255:
                grid[i][j] = 0
                continue    # Move on if we have a white cell
            
            # Now we are quite certain that this crop_image contains a number

            # Store the number of rows and cols
            rows, cols = crop_image.shape

            # Apply Binary Threshold to make digits more clear
            _, crop_image = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY) 
            crop_image = crop_image.astype(np.uint8)

            # Centralize the image according to center of mass
            crop_image = cv2.bitwise_not(crop_image)
            shift_x, shift_y = get_best_shift(crop_image)
            shifted = shift(crop_image, shift_x, shift_y)
            crop_image = shifted

            crop_image = cv2.bitwise_not(crop_image)
            
            # Up to this point crop_image is good and clean!
            #cv2.imshow(str(i)+str(j), crop_image)

            # Convert to proper format to recognize
            crop_image = prepare(crop_image)

            # Recognize digits
            prediction = model.predict([crop_image]) # model is trained by digitRecognition.py
            grid[i][j] = np.argmax(prediction[0]) + 1 # 1 2 3 4 5 6 7 8 9 starts from 0, so add 1
    return grid


# Calculate how to centralize the image using its center of mass
def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

# Shift the image using what get_best_shift returns
def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


# Prepare and normalize the image to get ready for digit recognition
def prepare(img_array):
    new_array = img_array.reshape(-1, 28, 28, 1)
    new_array = new_array.astype('float32')
    new_array /= 255
    return new_array


def load_model():
    # Loading model (Load weights and configuration seperately to speed up model and predictions)
    input_shape = (28, 28, 1)
    num_classes = 9
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Load weights from pre-trained model. This model is trained in digitRecognition.py
    model.load_weights("digitRecognition.h5") 
    return model


# This function is used for seperating the digit from noise in "crop_image"
# The Sudoku board will be chopped into 9x9 small square image,
# each of those image is a "crop_image"
def largest_connected_component(image):

    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    if (len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image

    max_label = 1
    # Start from component 1 (not 0) because we want to leave out the background
    max_size = sizes[1]     

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2
