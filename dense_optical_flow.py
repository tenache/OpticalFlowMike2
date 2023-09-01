import numpy as np 
import argparse 
import cv2 
from ultralytics import YOLO
from auxiliaries import get_edges
from auxiliaries import get_edges_from_cap
import os


parser = argparse.ArgumentParser(description="video and stuff")
parser.add_argument("-v", "--video", type=str, help= "path to video", default=os.path.join("sample_data2/short_video.dav"))
parser.add_argument("-m", "--model_path", type=str, help="path to detection model", default="yolov5nu.pt")
parser.add_argument('-d', "--detection_threshold", type=int, help ="How stringer should the detector be", default=0.4)
args = parser.parse_args()
detection_threshold = int(args.detection_threshold)
print(f"args.video is {args.video}")
cap = cv2.VideoCapture(args.video)

model = YOLO(args.model_path)
# get first video frame
ok, frame = cap.read()
print(f"frame.shape is {frame.shape}")
print(f"frame is {frame}")

# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
# TODO: switch for YOLO

parameters_shitomasi = dict(maxCorners=25, qualityLevel=0.3, minDistance=7)

# convert to grayscale
frame_gray_init = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# Use Shi-Tomasi to detect object corners / edges from initial frame
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **parameters_shitomasi)
print(f"edges shape is {edges.shape}")
print(f"edges is {edges}")
print(f"edges type is {type(edges)}")
detection_threshold = 0.4

result = model(frame)[0]
edges2 = []


edges2 = get_edges(result, edges2, detection_threshold=detection_threshold)
counter = 0  
frame_number = 0

# TODO: handle exception where no detections are made 
edges2, frame_number = get_edges_from_cap(cap=cap, model=model, frame_number=frame_number, edges2=edges2, detection_threshold=detection_threshold)
    
        
print(f"edges2 is {edges2}")
print(f"edges2 type is {type(edges2)}")
print(f"type of edges2[0][0][0] is {type(edges2[0][0][0])}")

# create a black canvas the size of the initial frame
canvas = np.zeros_like(frame)
# create random colours for visualization for all 100 max corners for RGB channels
colours = np.random.randint(0, 255, (100, 3))

# set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



# create canvas to paint on
hsv_canvas = np.zeros_like(frame)
# set saturation value (position 2 in HSV space) to 255
hsv_canvas[..., 1] = 255

edges2, frame_number = get_edges_from_cap(cap=cap, model=model, frame_number=frame_number, edges2=edges2, detection_threshold=detection_threshold)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = cap.read()

while True:
    # get next frame
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] reached end of file")
        break

    frame_gray = cv2.cvtColor(frame, cv2. COLOR_BGR2GRAY)
    
    # compare initial frame with current frame
    flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    # get x and y coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # set hue of HSV canvas (position 1)
    hsv_canvas[..., 0] = angle*(180/np.pi/2)
    # set pixel intensity value (position 3)
    hsv_canvas[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)
    
    # optional recording result/mask
    # video_output.write(frame_rgb)
    
    cv2.imshow('Optical Flow (dense)', frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # set initial frame to current frame
    frame_gray_init = frame_gray     