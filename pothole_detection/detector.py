
from typing import List, Dict, Any
from ultralytics import YOLO
import numpy as np
from config import settings


class PotholeDetector:

    def __init__(
        self,
        model_path: str = settings.MODEL_PATH,
        conf: float = settings.CONFIDENCE_THRESHOLD,
    ):

        self.model = YOLO(model_path)
        self.conf = conf

        print(f"[OK] Pothole model loaded: {model_path}")
        print("[OK] ByteTrack enabled")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track potholes using YOLO + ByteTrack.
        """

        detections: List[Dict[str, Any]] = []

        results = self.model.track(
            frame,
            conf=self.conf,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        if results.boxes is None:
            return detections

        for box in results.boxes:

            # ------------------------------------------------
            # Skip detection if ByteTrack has not assigned ID
            # ------------------------------------------------
            if box.id is None:
                continue

            xyxy = box.xyxy[0].cpu().numpy().tolist()

            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())

            # ByteTrack unique ID
            track_id = int(box.id[0].item())

            label = self.model.names.get(
                class_id,
                "pothole"
            )

            xmin, ymin, xmax, ymax = map(int, xyxy)

            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            detections.append(
                {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": confidence,
                    "class_id": class_id,
                    "label": label,
                    "centroid": (center_x, center_y),

                    # Unique ByteTrack ID
                    "track_id": track_id,
                }
            )

        return detections

