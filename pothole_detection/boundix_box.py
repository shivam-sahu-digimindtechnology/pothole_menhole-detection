
from typing import List, Dict, Any
import cv2
import numpy as np


BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)



def draw_bounding_boxes(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:

    annotated = frame.copy()

    for det in detections:

        xmin, ymin, xmax, ymax = det["bbox"]

        label = det.get("label", "pothole")
        confidence = det.get("confidence", 0.0)
        track_id = det.get("track_id", -1)

        # ID + confidence
        text = f"Pothole ID:{track_id} {confidence:.2f}"

        # Bounding box
        cv2.rectangle(
            annotated,
            (xmin, ymin),
            (xmax, ymax),
            BOX_COLOR,
            thickness
        )

        # Text size
        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1
        )

        text_y1 = max(0, ymin - th - baseline - 4)

        # Text background
        cv2.rectangle(
            annotated,
            (xmin, text_y1),
            (xmin + tw + 4, ymin),
            BOX_COLOR,
            -1
        )

        # Text
        cv2.putText(
            annotated,
            text,
            (xmin + 2, ymin - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            1,
            cv2.LINE_AA
        )

        # Centroid
        if "centroid" in det:

            cx, cy = det["centroid"]

            cv2.circle(
                annotated,
                (cx, cy),
                4,
                BOX_COLOR,
                -1
            )

    return annotated


def print_detections(
    detections: List[Dict[str, Any]],
    frame_idx: int = -1
) -> None:

    if not detections:
        return

    prefix = f"[Frame {frame_idx}] " if frame_idx >= 0 else ""

    print(
        f"{prefix}=== Pothole Detections "
        f"({len(detections)}) ==="
    )

    for i, det in enumerate(detections, 1):

        bbox = det["bbox"]

        print(
            f"  {i}. "
            f"ID={det.get('track_id', '-'):<4} "
            f"{det.get('label', 'pothole'):10s} "
            f"conf={det.get('confidence', 0):.3f} "
            f"bbox=[{bbox[0]}, {bbox[1]}, "
            f"{bbox[2]}, {bbox[3]}]"
        )

    print()

