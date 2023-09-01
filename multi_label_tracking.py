#track multiple objects and not meant to run on RPI without hardware acceleration
import numpy as np
import argparse
import cv2 
import time
from imutils.video import FPS 
import os

# construct the argument parse 
parser = argparse.ArgumentParser(description='Script to run Object trackers using opencv')
parser.add_argument("-v", "--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--thr", default=0.6, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--frame_count", default='5',help="run the object detector every n frames")
parser.add_argument("--output",default = False,help = "create output video file")
args = parser.parse_args()

# Labels of Network.
labels = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

lk_params = dict(winSize = (50,50), maxLevel = 4, 
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
    base_name, ext = os.path.splitext(args.video)
    if base_name:
        output_file = base_name + ".mp4"
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define the code using VideoWriter_fourcc and create VideoWriter object
        fourcc =cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
else:
    cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe("mobilenet/MobileNetSSD_deploy.prototxt", "mobilenet/MobileNetSSD_deploy.caffemodel")
fps = FPS().start()
total_frames = 1
_, prev_frame = cap.read()
tracking_started = False

if args.output:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args.output, fourcc, 30,(prev_frame.shape[1], prev_frame.shape[0]), True)

counter = 0
while True:
    # time.sleep(0.25)
    # counter += 1
    # print(f"frame is about {counter}")
    _,frame = cap.read()
    if frame is None: #end of video file
        break
    frame_resized = cv2.resize(frame,(300,300)) # reshaping frame to (300,300)
    # running the object detector every nth frame 
    if total_frames % int(args.frame_count)-1 == 0:

        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, 
            (frame_resized.shape[1],frame_resized.shape[0]), (127.5, 127.5, 127.5), crop = False)
        net.setInput(blob)
        detections = net.forward()
        # print(detections)
        # object to be tracked's probability should be greater than the threshold
        idx = np.argwhere(detections[0, 0, :, 2] >= args.thr)
        centroids = np.zeros([1, 1, 2], dtype=np.float32)

        for i in range(0,len(idx)):     
            tracking_id = int(detections[0, 0, idx[i], 1]) 
            confidence = detections[0, 0, idx[i], 2]

            # Object location 
            xLeftBottom = int(detections[0, 0, idx[i], 3] * frame_resized.shape[1]) 
            yLeftBottom = int(detections[0, 0, idx[i], 4] * frame_resized.shape[0])
            xRightTop   = int(detections[0, 0, idx[i], 5] * frame_resized.shape[1])
            yRightTop   = int(detections[0, 0, idx[i], 6] * frame_resized.shape[0])
            
            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/frame_resized.shape[0]
            widthFactor = frame.shape[1]/frame_resized.shape[1]

            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)

            # print class and confidence
            label = labels[tracking_id] +": "+ str(confidence)             
            print(label) 

            x = (xLeftBottom + xRightTop)/2
            y = (yLeftBottom + yRightTop)/2

            # draw the centroid on the frame
            frame = cv2.circle(frame, (int(x),int(y)), 15, (0,0,255), -1)
            tracking_started = True
            if i == 0:
                centroids[0,0,0] = x
                centroids[0,0,1] = y
            else:
                centroid = np.array([[[x,y]]],dtype=np.float32)
                centroids = np.append(centroids,centroid,axis = 0)

    else:   # track an object only if it has been detected
        if centroids.sum() != 0 and tracking_started:
            next1, st, error = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                            centroids, None, **lk_params)

            good_new = next1[st==1]
            good_old = centroids[st==1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # Returns a contiguous flattened array as (x, y) coordinates for new point
                a, b = new.ravel()
                c, d = old.ravel()
                distance = np.sqrt((a-c)**2 + (b-d)**2)
                # distance between new and old points should be less than
                # 200 for 2 points to be same the object
                if distance < 200 :
                    frame = cv2.circle(frame, (int(a), int(b)), 15, (0,0,255), -1)

            centroids = good_new.reshape(-1, 1, 2)

    total_frames += 1
    fps.update()
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    if args.output:
        writer.write(frame)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    prev_frame = frame
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break