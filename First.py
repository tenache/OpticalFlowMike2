import argparse
import cv2
import numpy as np
import os
from ultralytics import YOLO
from time import sleep
from auxiliaries import get_edges
from auxiliaries import get_edges_from_cap
from auxiliaries import get_first_det_frame 
from trackableObject import CentroidTracker, TrackableObject

## TODO: fijarse como hacer para que no tire error cuando el YOLO no lo agarra
## TODO: fijarse como hacer para que trabaje con edges y no con centroides. 
## TODO: create git repository
## TODO: make an app


## TODO: look in huggingface

parser = argparse.ArgumentParser(description="video and stuff")
parser.add_argument("-v", "--video", type=str, help= "path to video", default=os.path.join("sample_data2/short_video.dav"))
parser.add_argument("-m", "--model_path", type=str, help="path to detection model", default="yolov5nu.pt")
args = parser.parse_args()

# print(f"args.video is {args.video}")
cap = cv2.VideoCapture(args.video)

model = YOLO(args.model_path)

# get first video frame where a person detection was made
ok, frame = get_first_det_frame(cap, model)

# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
# TODO: switch for YOLO

parameters_shitomasi = dict(maxCorners=25, qualityLevel=0.3, minDistance=7)

# convert to grayscale
frame_gray_init = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)



detection_threshold = 0.4

result = model(frame)[0]


# Use Shi-Tomasi to detect object corners / edges from initial frame
# Detect Shi-Tomasi corners only in rectangles perceived by YOLO model 
# gets edges with the YOLO model 
# For now, it just gets the centroid of the image, but its possible to detect more edges 
edges2 = []
edges_as_box = []
# edges2, edges_as_box = get_edges(result, edges2, edges_as_box, detection_threshold=detection_threshold)
counter = 0  
frame_number = 0

# mask = np.zeros_like(frame_gray_init)
# print(f"mask.shape is {mask.shape}")
# for edge in edges_as_box:
#     print('heyho')
#     sleep(1)
#     print(edge)
#     x1, y1, x2, y2 = edge
#     mask[x1:x2,y1:y2] = 1
    
# print(f"sum of mask is {sum(mask)}")
# print(f"sum of sum of mask is {(sum(sum(mask)))}")

# edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **parameters_shitomasi)


# # TODO: handle exception where no detections are made 
edges2, frame_number, edges_as_box = get_edges_from_cap(cap=cap, model=model, frame_number=frame_number, edges2=edges2, detection_threshold=detection_threshold)

# get mask of our region of interest. Only get edges fromt there
mask = np.zeros_like(frame_gray_init)
print(f"edges as box is {edges_as_box}")

for edge in edges_as_box:
    print('heyho')
    sleep(1)
    print(edge)
    x1, y1, x2, y2 = edge
    mask[y1:y2,x1:x2] = 255

print(mask)
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=mask, **parameters_shitomasi)

edges = edges.astype(np.float32)



# create a black canvas the size of the initial frame
canvas = np.zeros_like(frame)
# create random colours for visualization for all 100 max corners for RGB channels
colours = np.random.randint(0, 255, (100, 3))
    

cv2.circle(frame_gray_init, (int(edges[0][0][0]), int(edges[0][0][1])), 20, (255, 0, 0), 2)
# cv2.imshow("Window name", frame_gray_init)
cv2.circle(frame, (int(edges[0][0][0]), int(edges[0][0][1])), 100, (255, 0, 0), 2)
cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (255, 0, 0), 2)

# cv2.imshow("window name", frame)
# cv2.waitKey(0)
# cv2.imshow("mask", mask)

# set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

max_frame_number = 0
ct = CentroidTracker(maxDisappeared=40)
trackers = []
TrackableObjects = {}
while True:
    frame_number += 1
    if frame_number > max_frame_number:
        max_frame_number = frame_number
    else:
        print(f"max_frame_number is {max_frame_number} and frame_number is {frame_number}")
        break
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    
    if not ok:
        print("[INFO] end of file reached")
        break
    # prepare grayscale image
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # update object corners by comparing with found edges in initial frame
    update_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges2, None, **parameter_lucas_kanade)
    
    # only update edges if algorithm successfully tracked
    new_edges = update_edges[status == 1]
    # to calculate directional flow we need to compare with previous position
    old_edges = edges2[status == 1]
    
    for i, (new, old) in enumerate(zip(new_edges, old_edges)):
        a, b = new.ravel()
        c, d = old.ravel()
        # draw line between old and new corner point with random colour
        mask = cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), colours[i].tolist(), 2)
        # draw circle around new position
        frame = cv2.circle(frame, (int(a), int(b)), 5, colours[i].tolist(), -1)
        
    result = cv2.add(frame, mask)
    cv2.imshow('Optical Flow (sparse)', result)
    # cv2.imshow('Window name', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # overwrite initial frame with current before restarting the loop
    frame_gray_init = frame_gray.copy()
    # update to new edge before restarting the loop
    edges2 = new_edges.reshape(-1, 1, 2)
    
    

    
    
  

    
    
    
    
