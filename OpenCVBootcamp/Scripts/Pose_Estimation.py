import sys
import cv2

protoFile = "OpenCVBootcamp/Models/pose_deploy_linevec_faster_4_stages.prototxt"
modelFile = "OpenCVBootcamp/Models/pose_iter_160000.caffemodel"
numberOfPoints = 15
posePairs = [[0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 14],
    [14, 8],
    [8, 9],
    [9, 10],
    [14, 11],
    [11, 12],
    [12, 13]
    ]
net = cv2.dnn.readNetFromCaffe(protoFile, modelFile)
dimension = (368, 368)

threshold = 0.1

source = cv2.VideoCapture(1)
cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
while cv2.waitKey(1) != 27:
    read_success, frame = source.read()
    if not read_success:
        break

    frame = cv2.flip(frame, 1)
    skeletonFrame = frame.copy()
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]

    inputBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, dimension, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inputBlob)  

    output = net.forward()

    scaleX = frameWidth / output.shape[3]
    scaleY = frameHeight / output.shape[2]

    points = []

    for i in range(numberOfPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (scaleX * point[0])
        y = (scaleY * point[1])

        if prob > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    for pair in posePairs:
        pairA = pair[0]
        pairB = pair[1]

        if points[pairA] and points[pairB]:
            cv2.line(skeletonFrame, points[pairA], points[pairB], (255, 255, 0), 2)
            cv2.circle(skeletonFrame, points[pairA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)


    cv2.imshow("Pose Estimation", skeletonFrame)

