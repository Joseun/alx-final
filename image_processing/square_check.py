#!/usr/bin/env python3
import cv2
import numpy as np
import math


# Return the angle between 2 vectors in degrees
def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector2)
    angle = np.arccos(dot_product)
    return angle * 57.2958  # Convert to degree


# This function is used as the second criteria for detecting whether
# the contour is a Sudoku board or not: All 4 angles has to be approximately 90 degree
# Approximately 90 degress with tolerance epsilon
def approx_90_degrees(angle, epsilon):
    return abs(angle - 90) < epsilon


def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest


def square_check(location, image):
    """
    Function to locate the top left, top right, bottom left
    and bottom right corners

    Arguments:
    corners: black and white camera image
    image: camera image

    Return: tuple of positions
    """
    A = location[0]
    B = location[1]
    C = location[2]
    D = location[3]
    
    # 1st condition: If all 4 angles are not approximately 90 degrees (with tolerance = epsAngle), stop
    AB = B - A      # 4 vectors AB AD BC DC
    AD = D - A
    BC = C - B
    DC = C - D
    eps_angle = 20
    if not (approx_90_degrees(angle_between(AB,AD), eps_angle) and approx_90_degrees(angle_between(AB,BC), eps_angle)
    and approx_90_degrees(angle_between(BC,DC), eps_angle) and approx_90_degrees(angle_between(AD,DC), eps_angle)):
        return image
    
    # 2nd condition: The Lengths of AB, AD, BC, DC have to be approximately equal
    # => Longest and shortest sides have to be approximately equal
    eps_scale = 1.2     # Longest cannot be longer than epsScale * shortest
    if (side_lengths_are_too_different(A, B, C, D, eps_scale)):
        return image

    perimeter = dimensions(location, image)
    return perimeter


def dimensions(location, image):
    # Now we are sure ABCD correspond to 4 corners of a Sudoku board

    # the width of our Sudoku board
    (tl, tr, br, bl) = location
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # the height of our Sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = "float32")
    perspective_transformed_matrix = cv2.getPerspectiveTransform(location, dst)
    warp = cv2.warpPerspective(image, perspective_transformed_matrix, (max_width, max_height))
    
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen

    # At this point, warp contains ONLY the chopped Sudoku board
    # Do some image processing to get ready for recognizing digits
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    warp = cv2.GaussianBlur(warp, (5, 5), 0)
    warp = cv2.adaptiveThreshold(warp, 255, 1, 1, 11, 2)
    warp = cv2.bitwise_not(warp)
    _, warp_image = cv2.threshold(warp, 150, 255, cv2.THRESH_BINARY)
    return [warp_image, perspective_transformed_matrix]
