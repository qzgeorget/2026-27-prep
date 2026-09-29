import cv2
import sys

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

source = cv2.VideoCapture(1)

window_name = "Face Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe("OpenCVBootcamp/Models/deploy.prototxt", "OpenCVBootcamp/Models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

in_width = 300
in_height = 300
mean_val = [104.0, 177.0, 123.0]
conf_threshold = 0.7

while cv2.waitKey(1) != 27:
    read_success, frame = source.read()
    if not read_success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[0], frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean_val, swapRB=False, crop=False)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top),(0, 255, 0))

            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),(x_left_bottom + label_size[0], y_left_bottom + base_line),(255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()
    inference_time = t * 1000.0 / cv2.getTickFrequency()
    label = "Inference time: %.2f ms" % inference_time
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imshow(window_name, frame)