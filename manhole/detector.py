from typing import List, Dict, Any
from ultralytics import YOLO
import numpy as np
from config import settings


class ManholeDetector:

    def __init__(
        self,
        model_path: str = settings.MODEL_PATH,
        conf: float = settings.CONFIDENCE_THRESHOLD,
    ):
        self.model = YOLO(model_path)
        self.conf = conf

        print(f"[OK] Manhole model loaded: {model_path}")
        print(f"[OK] Classes: {self.model.names}")
        print("[OK] ByteTrack enabled")

    def _get_status(self, label: str) -> str:
        """
        Map model class name to status: 'open' or 'closed'
        Your model classes:
          - "Open Manhole"  → open
          - "Manhole"       → closed
        """
        label_lower = label.lower().strip()

        if "open" in label_lower:
            return "open"
        else:
            # "Manhole" (without open) = closed cover
            return "closed"

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track manholes (open / closed) using YOLO + ByteTrack.
        """
        detections: List[Dict[str, Any]] = []

        results = self.model.track(
            frame,
            conf=self.conf,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        if results.boxes is None:
            return detections

        for box in results.boxes:

            if box.id is None:
                continue

            xyxy = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            track_id = int(box.id[0].item())

            # Original class name from model (e.g. "Open Manhole", "Manhole")
            raw_label = self.model.names.get(class_id, "Manhole")
            status = self._get_status(raw_label)  # "open" or "closed"

            xmin, ymin, xmax, ymax = map(int, xyxy)
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            detections.append(
                {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": confidence,
                    "class_id": class_id,
                    "label": raw_label,       # show exact name like image
                    "status": status,         # open / closed for counting
                    "centroid": (center_x, center_y),
                    "track_id": track_id,
                }
            )

        return detections