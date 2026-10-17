from sahi.utils.yolov8 import (
    download_yolov8s_model
)
import cv2
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import os







def detee(video_path, model_path="./best5.pt", confidence_threshold=0.3, device='cpu'):
    detection_model = AutoDetectionModel.from_pretrained(
         model_type= 'yolov8',
         model_path= model_path,
         confidence_threshold= confidence_threshold,
         device= device
        )
    img = read_first_frame(video_path)
    result = get_sliced_prediction(img, detection_model, slice_width=1024, slice_height=1024, overlap_height_ratio=0.25, overlap_width_ratio=0.25)
    # Assuming result2.to_coco_annotations() is stored in the variable 'annotations'
    annotations = result.to_coco_annotations()
    # Extract bbox coordinates with id
    bbox_coordinates = [{"id": i, "bbox": item['bbox']} for i, item in enumerate(annotations)]
    print(bbox_coordinates)
    return bbox_coordinates


def read_first_frame(video_path):
    """
    Reads the first frame of a video and returns it as an image.

    Args:
        video_path (str): Path to the video file.

    Returns:
        img (numpy.ndarray): The first frame of the video as an image.
        None: If the frame could not be read or the video could not be opened.
    """
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    last_frame = None
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if (not ret):
    #         break
    #     try:
    #         last_frame = frame
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         break

    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)  # Wait until any key is pressed
    
    # cap.release()
    
    cap = cv2.VideoCapture(video_path)
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Read the first frame
    ret, img = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        img = None

    # Release the VideoCapture object
    cap.release()
    img = cv2.imread(r'Data\Pictures\VideoCapture_20240608-224744.jpg')
    return img