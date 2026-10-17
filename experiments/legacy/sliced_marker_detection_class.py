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


class My():
    def __init__(self):
        print("Hello")
        
def detee(model_path="./best5.pt", confidence_threshold=0.15, device='cpu', image_frame = None):
    detection_model = AutoDetectionModel.from_pretrained(
         model_type= 'yolov8',
         model_path= model_path,
         confidence_threshold= confidence_threshold,
         device= device
        )
    img = image_frame
    result = get_sliced_prediction(img, detection_model, slice_width=1024, slice_height=1024, overlap_height_ratio=0.25, overlap_width_ratio=0.25)
    # Assuming result2.to_coco_annotations() is stored in the variable 'annotations'
    annotations = result.to_coco_annotations()
    # Extract bbox coordinates with id
    bbox_coordinates = [{"id": i, "bbox": item['bbox']} for i, item in enumerate(annotations)]
    print(bbox_coordinates)
    return bbox_coordinates
        
class PlatesDetection():
    def __init__(self, model_path="./best5.pt", confidence_threshold=0.15, device='cpu'):

         # Construct the full path to the model file
        self.model_path = os.path.abspath(model_path)  # Use absolute path

        # Debug: Print the model path
        print(f"Model path: {self.model_path}")
        if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
        self.model_path = os.path.join("subfolder", model_path)  # Construct full path

        self.detection_model = AutoDetectionModel.from_pretrained(
         model_type= 'yolov8',
         model_path= self.model_path,
         confidence_threshold= confidence_threshold,
         device= device
        )
        
    def detect(self, image_frame):
        self.img = image_frame
        self.result = get_sliced_prediction(self.img, self.detection_model, slice_width=1024, slice_height=1024, overlap_height_ratio=0.25, overlap_width_ratio=0.25)
        # Assuming result2.to_coco_annotations() is stored in the variable 'annotations'
        annotations = self.result.to_coco_annotations()

        # Extract bbox coordinates with id
        bbox_coordinates = [{"id": i, "bbox": item['bbox']} for i, item in enumerate(annotations)]

        print(bbox_coordinates)
        return bbox_coordinates
    
    