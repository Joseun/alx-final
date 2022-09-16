#!/usr/bin/env python3
import cv2
import numpy as np
import math
from . import sudoku_solver
import copy


def solve_image(frame, old_sudoku, grid, processed_image, transformed_matrix):
    user_grid = copy.deepcopy(grid)
    image = copy.deepcopy(frame)
    # Solve sudoku after we have recognizing each digits of the Sudoku board:

    # If this is the same board as last camera frame
    # Phewww, print the same solution. No need to solve it again
    if (not old_sudoku is None) and two_matrices_are_equal(old_sudoku, grid, 9, 9):
        if(all_board_non_zero(grid)):
            solved_image = write_solution_on_image(processed_image, old_sudoku, user_grid)
    # If this is a different board
    else:
        sudoku_solver.solve_sudoku(grid) # Solve it
        if(all_board_non_zero(grid)): # If we got a solution
            solved_image = write_solution_on_image(processed_image, grid, user_grid)
            old_sudoku = copy.deepcopy(grid)      # Keep the old solution

    # Apply inverse perspective transform and paste the solutions on top of the orginal image
    result_sudoku = cv2.warpPerspective(solved_image, transformed_matrix, (image.shape[1], image.shape[0])
                                        , flags=cv2.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1,keepdims=True)!=0, result_sudoku, image)

    return result


# Write solution on "image"
def write_solution_on_image(image, grid, user_grid):
    # Write grid on image
    SIZE = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if(user_grid[i][j] != 0):    # If user fill this cell
                continue                # Move on
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
            marginX = math.floor(width / 7)
            marginY = math.floor(height / 7)
        
            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + off_set_y
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), 
                                                  font, font_scale, (0,0,255), thickness=3, lineType=cv2.LINE_AA)
    return image


# Compare every single elements of 2 matrices and return if all corresponding entries are equal
def two_matrices_are_equal(matrix_1, matrix_2, row, col):
    for i in range(row):
        for j in range(col):
            if matrix_1[i][j] != matrix_2[i][j]:
                return False
    return True


# Return true if the whole board has been occupied by some non-zero number
# If this happens, the current board is the solution to the original Sudoku
def all_board_non_zero(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True
