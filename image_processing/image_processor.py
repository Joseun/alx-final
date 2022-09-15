#!/usr/bin/env python3
""" Module for detecting a Sudoku puzzle """
import cv2
import numpy as np
from . import square_check


def convert_image(image):
    """
    Function to adjust sudoku sheet from the camera to black background

    Arguments:
    image: camera image

    Return: adjusted image
    """
    # Convert image to a grayscale,
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # blur that gray image for easier detection
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # apply adaptiveThreshold
    adapted_image = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    final_image = find_contours(adapted_image, image)

    return final_image


def find_contours(adapted_image, image):
    """
    Function to find the area spanned by the sheet

    Arguments:
    adapted_image: black and white camera image
    image: camera image

    Return: corners or image if failed
    """
    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the puzzle image has the biggest contour,
    # extract the contour with the biggest area
    max_area = 0
    max_contour = None
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > max_area:
            max_area = contour_area
            max_contour = contour

    if max_contour is None:
        return image

    image_with_corners = corners_from_contours(max_contour, image)
    return image_with_corners


def corners_from_contours(contours, image):
    """
    Function to find the corners of the sheet

    Arguments:
    contours: the area covered by the sheet

    Return: corners or image if failed
    """
    coefficient = 1
    max_iter = 200
    while max_iter > 0 and coefficient >= 0:
        max_iter -= 1

        epsilon = coefficient * cv2.arcLength(contours, True)

        poly_approx = cv2.approxPolyDP(contours, epsilon, True)
        corners = cv2.convexHull(poly_approx)
        if len(corners) == 4:
            image_with_4edges = orient_corners(corners)
            return image_with_4edges
        else:
            if len(corners) > corner_amount:
                coefficient += .01
            else:
                coefficient -= .01
    return image


def orient_corners(corners):
    """
    Function to locate the top left, top right, bottom left
    and bottom right corners

    Arguments:
    corners: black and white camera image
    image: camera image

    Return: tuple of positions
    """
    location = np.zeros((4, 2), dtype="float32")
    corners = corners.reshape(4, 2)

    # Find top left (sum of coordinates is the smallest)
    sum = 10000
    index = 0
    for i in range(4):
        if (corners[i][0] + corners[i][1] < sum):
            sum = corners[i][0] + corners[i][1]
            index = i
    location[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find bottom right (sum of coordinates is the biggest)
    sum = 0
    for i in range(3):
        if (corners[i][0] + corners[i][1] > sum):
            sum = corners[i][0] + corners[i][1]
            index = i
    location[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find top right (Only 2 points left)
    if(corners[0][0] > corners[1][0]):
        location[1] = corners[0]
        location[3] = corners[1]
    else:
        location[1] = corners[1]
        location[3] = corners[0]

    location = location.reshape(4, 2)
    grid_perimeter = square_check.square_check(location, image)
    processed_image = perpective_grab(image, location, grid_perimeter)
    return processed_image


def perpective_grab(image, location, dist):
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    perspective_transformed_matrix = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, perspective_transformed_matrix,
                               (max_width, max_height))
    warp_copy = np.copy(warp)

    # At this point, warp contains ONLY the chopped Sudoku board
    # Do some image processing to get ready for recognizing digits
    warp_copy = cv2.cvtColor(warp_copy, cv2.COLOR_BGR2GRAY)
    warp_copy = cv2.GaussianBlur(warp_copy, (5, 5), 0)
    warp_copy = cv2.adaptiveThreshold(warp_copy, 255, 1, 1, 11, 2)
    warp_copy = cv2.bitwise_not(warp_copy)
    _, warp_image = cv2.threshold(warp_copy, 150, 255, cv2.THRESH_BINARY)
    return warp_image
