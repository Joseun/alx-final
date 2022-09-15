#!/usr/bin/env python3
""" Module for solving a Sudoku puzzle in real time """
import cv2
from image_processing.image_processor import convert_image
from image_processing.image_to_grid import image_to_grid
from sudoku_solver.solve_image import solve_image


# Load and set up Camera to grab puzzle
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)    # HD Camera
cap.set(4, 720)

# Let's turn on webcam
ret, frame = cap.read() # Read the frame
old_sudoku = None
while(True):
    if ret == True:
        # process the image to get a virtual grid with digits
        processed_image = convert_image(frame)
        sheet = image_to_grid(processed_image)
        sudoku_frame = solve_image(frame, old_sudoku, sheet, processed_image)
        showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600) # Print the 'solved' image
        if cv2.waitKey(1) & 0xFF == ord('q'):   # Hit q if you want to stop the camera
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

print("done")