import cv2 as cv
import sys

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

source = cv.VideoCapture(s)

win_name = "Webcam Feed"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

while cv.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        print("No frames grabbed!")
        break
    cv.imshow(win_name, frame)

source.release()
cv.destroyWindow(win_name)