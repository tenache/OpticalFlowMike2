import cv2
import numpy as np

def get_first_det_frame(cap, model):
    ok, frame = cap.read()
    result = model(frame)[0]
    while len(result.boxes.data.tolist()) < 1:
        ok, frame = cap.read()
        result = model(frame)[0]
        if not ok:
            raise Exception("No detections in this video")
    return ok, frame
    

def get_edges(result, edges2,edges_as_box, detection_threshold):
    # edges2 = list(edges2)
    # edges_as_box = list(edges_as_box)
    len_edges = len(edges2)
    len_edges_as_box = len(edges_as_box)
    for i, r in enumerate(result.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = r
        if class_id == 0 and score > detection_threshold:
            edges2.append(np.array(((x1+x2)/2, (y1+y2)/2)))
            edges_as_box.append(np.array([int(x1), int(y1), int(x2), int(y2)]))
            # edges2.append(np.array(((x1+x2)/2, (y1+y2)/2)))
            # edges_as_box.append([int(x1), int(y1), int(x2), int(y2)])
    return edges2, edges_as_box


def get_edges_from_cap(cap, model, frame_number, edges2,detection_threshold, started=False): 
    edges2 = []
    edges_as_box = []
    while len(edges2) < 1: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        result = model(frame)[0]
        edges2, edges_as_box = get_edges(result, edges2, edges_as_box, detection_threshold)
        frame_number += 1
    # edges are points
    edges2 = np.array(edges2)
    edges_as_box = np.array(edges_as_box)
    edges2 = edges2.reshape(-1, 1, 2).astype(np.float32)
    edges_as_box = edges_as_box.astype(int)

    return edges2, frame_number, edges_as_box