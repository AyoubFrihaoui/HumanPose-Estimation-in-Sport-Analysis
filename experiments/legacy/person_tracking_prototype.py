import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque


model = YOLO('yolov8s.pt')  # You can replace 'yolov8s.pt' with the path to your YOLOv8 model
video_path = None
def main(video_path):
    video_path = video_path
    model = YOLO('yolov8s.pt')
    bbox, id, tracks = process_video(video_path)
    velocities = {}
    for i in tracks:
        box, v = calculate_max_horizontal_velocity(tracks[i][2])
        velocities[i] = v

    max_velocity_id, max_velocity = find_max_velocity(velocities)
    print(max_velocity_id, max_velocity)
    print(tracks.keys())
    print(tracks[max_velocity_id][2], max_velocity)
    
    


def detect_persons(frame):
    model = YOLO('yolov8s.pt')
    results = model(frame, conf = 0.1)
    person_boxes = []
    
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        cls = int(result.cls)
        if cls == 0:  # Class ID 0 is for 'person'
            person_boxes.append((x1, y1, x2 - x1, y2 - y1))
    
    return person_boxes

def calculate_horizontal_velocity(current_positions, previous_positions):
    velocities = {}
    for i in current_positions:
        if i in previous_positions:
            velocities[i] = abs(current_positions[i][0] - previous_positions[i][0])
        
    return velocities



def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    #tracker = cv2.TrackerKCF_create()
    prev_positions = {}
    max_velocity = 0
    bbox_highest_velocity = None

    tracks = {}
    track_id = 0
    max_velocity_track_id = None
    frames_buffer = 6000  # Number of frames to keep for each track
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons in the frame
        person_boxes = detect_persons(frame)
        

        # Track each person
        current_positions = {}
        for box in person_boxes:
            x, y, w, h = box
            center_x = x + w / 2
            center_y = y + h / 2
        

            # Find if this is a new track or existing one
            found = False
            for trackid, (prev_center_x, prev_center_y, track) in tracks.items():
                if abs(center_x - prev_center_x) < w and abs(center_y - prev_center_y) < h:
                    print("changed in track id ", trackid, )
                    found = True
                    current_positions[trackid] = (center_x, box)
                    tracks[trackid] = (center_x, center_y, track + [[x,y,w,h]])
                    break

            if not found:
                temp = list([[x,y,w,h]])
                tracks[track_id] = (center_x, center_y, temp)#deque(maxlen=frames_buffer))
                current_positions[track_id] = (center_x, box)
                print("track id ", track_id)
                track_id += 1
                

        velocities = calculate_horizontal_velocity(current_positions, prev_positions)
        for trackid, velocity in velocities.items():
            if velocity > max_velocity:
                print("happened ", velocity, " track id  ", trackid)
                max_velocity = velocity
                max_velocity_track_id = trackid
                bbox_highest_velocity = current_positions[trackid][1]

        prev_positions = current_positions
        

            
        i += 1

        # Draw bounding boxes and track IDs
        for track_id, (center_x, center_y, track) in tracks.items():
            if track_id in current_positions:
                box = current_positions[track_id][1]
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        #cv2.imshow('Frame', cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return bbox_highest_velocity, max_velocity_track_id, tracks

# Example usage
#video_path = "path_to_your_video.mp4"
bbox, id, tracks = process_video(video_path)


def calculate_max_horizontal_velocity(dataset):
    if not dataset:
        return None

    # Extract all x-coordinates
    x_coords = [box[0] for box in dataset]

    # Calculate max and min x-coordinates
    max_x = max(x_coords)
    min_x = min(x_coords)

    # Calculate horizontal velocity
    horizontal_velocity = max_x - min_x

    # Find the bounding box with the max x-coordinate
    max_x_box = next(box for box in dataset if box[0] == max_x)

    return max_x_box, horizontal_velocity


def find_max_velocity(velocity_dict):
    if not velocity_dict:
        return None

    # Find the ID with the maximum velocity
    max_id = max(velocity_dict, key=velocity_dict.get)
    max_velocity = velocity_dict[max_id]

    return max_id, max_velocity