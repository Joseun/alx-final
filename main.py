# # import the opencv library
# import cv2
  
  
# # define a video capture object
# vid = cv2.VideoCapture(0)
  
# while(True):
      
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
  
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
      
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
  
# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()

#!/usr/bin/env python3
""" Module for solving a Sudoku puzzle in real time """
import cv2
import numpy as np
from image_processing.image_processor import convert_image
from image_processing.image_to_grid import image_to_grid
from sudoku_solver.solve_image import solve_image


def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

# Load and set up Camera to grab puzzle
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)    # HD Camera
cap.set(4, 720)

# Let's turn on webcam
old_sudoku = None
while(True):
    ret, frame = cap.read() # Read the frame
    if ret == True:
        # process the image to get a virtual grid with digits
        processed_image_matrix = convert_image(frame)
        sheet = image_to_grid(processed_image_matrix[0])
        sudoku_frame = solve_image(frame, old_sudoku, sheet, processed_image_matrix[0], processed_image_matrix[1])
        showImage(processed_image_matrix[0], "Real Time Sudoku Solver", 1066, 600) # Print the 'solved' image
        if cv2.waitKey(1) & 0xFF == ord('q'):   # Hit q if you want to stop the camera
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
