from ultralytics import YOLO
import numpy as np

class YOLOSegmentation:
    def init(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        # Get img shape
        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        bboxes = []
        class_ids = []
        scores = []
        for i, seg in enumerate(result.masks.segments):
            # contours
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

            bbox = np.array(result.boxes.xyxy[i].cpu(), dtype="int")
            bboxes.append(bbox)
            # Get class ids
            class_id = np.array(result.boxes.cls[i].cpu(), dtype="int")
            class_ids.append(class_id)
            # Get scores
            score = np.array(result.boxes.conf[i].cpu(), dtype="float").round(2)
            scores.append(score)

        return bboxes, class_ids, segmentation_contours_idx, scores