import cv2 as cv
import numpy as np
import sys

# Outlining the different modes for the user to control while the window is opened.
PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

# Declaring and setting up the webcam/phone connection.
PHONE = 0
WEBCAM = 1
source_id = PHONE

image_filter = PREVIEW
alive = True

window_name = "Camera Filter"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
result = None

source = cv.VideoCapture(source_id)

# While-loop to react to use input while the window is opened.
while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    if source_id == WEBCAM:
        frame = cv.flip(frame, 1)

    if image_filter == PREVIEW: # Simple image preview
        result = frame
    elif image_filter == BLUR: # Box blur
        result = cv.blur(frame, (13, 13))
    elif image_filter == CANNY: # Canny edge detection
        result = cv.Canny(frame, 145, 150)
    elif image_filter == FEATURES: # Feature detection using Shi-Tomasi corner detection
        result = frame 
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(
            image=frame_gray,
            maxCorners=500,
            qualityLevel=0.2,
            minDistance=15,
            blockSize=9
        )
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                cv.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

    if result is not None:
        cv.imshow(window_name, result)

    # Controls 
    key = cv.waitKey(1)
    if key == ord('q') or key == ord('Q') or key == 27:
        alive = False
    elif key == ord('c') or key == ord('C'):
        image_filter = CANNY
    elif key == ord('b') or key == ord('B'):
        image_filter = BLUR
    elif key == ord('f') or key == ord('F'):
        image_filter = FEATURES
    elif key == ord('p') or key == ord('P'):
        image_filter = PREVIEW
    elif key == ord('t') or key == ord('T'): # Toggle between webcam and phone camera, might not be the most efficient way to do this, but it works for now.
        if source_id == PHONE:
            source.release()
            source_id = WEBCAM
            source = cv.VideoCapture(source_id)
        else:
            source.release()
            source_id = PHONE
            source = cv.VideoCapture(source_id)

source.release()
cv.destroyWindow(window_name)

