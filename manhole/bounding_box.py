from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


# Colors (BGR) - same green style as your image result
COLOR_OPEN = (0, 255, 0)        # green for Open Manhole
COLOR_CLOSED = (0, 255, 0)      # green for Manhole (closed)
COLOR_DEFAULT = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)          # black text like your image
TEXT_BG = (0, 255, 0)           # green background


def _get_color(label: str) -> Tuple[int, int, int]:
    # Keep same green style as your working image result
    return COLOR_OPEN


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw boxes exactly like your image result:
    Green box + "Open Manhole 86%" / "Manhole 89%"
    """
    annotated = frame.copy()

    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        label = det.get("label", "Manhole")          # "Open Manhole" or "Manhole"
        confidence = det.get("confidence", 0.0)
        track_id = det.get("track_id", -1)

        color = _get_color(label)

        # Same style as your image: "Open Manhole 86%"
        text = f"{label} {confidence:.0%}"

        # Bounding box
        cv2.rectangle(
            annotated,
            (xmin, ymin),
            (xmax, ymax),
            color,
            thickness,
        )

        # Text size
        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )

        text_y1 = max(0, ymin - th - baseline - 6)

        # Green filled background for text (like your image)
        cv2.rectangle(
            annotated,
            (xmin, text_y1),
            (xmin + tw + 4, ymin),
            color,
            -1,
        )

        # Black text on green background
        cv2.putText(
            annotated,
            text,
            (xmin + 2, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_stats_overlay(
    frame: np.ndarray,
    total: int,
    open_count: int,
    closed_count: int,
    frame_idx: int = 0,
    current_open: int = 0,
    current_closed: int = 0,
) -> np.ndarray:
    """
    Overlay on top-left:
      Frame
      Total Manholes
      Open
      Closed
      Current frame counts
    """
    lines = [
        (f"Frame: {frame_idx}", (0, 255, 255)),
        (f"Total Manholes : {total}", (255, 255, 255)),
        (f"Open           : {open_count}", (0, 0, 255)),
        (f"Closed         : {closed_count}", (0, 255, 0)),
        (f"Current -> Open: {current_open}  Closed: {current_closed}", (255, 255, 0)),
    ]

    y = 30
    for text, color in lines:
        # Black outline for readability
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        y += 32

    return frame


def print_detections(
    detections: List[Dict[str, Any]],
    frame_idx: int = -1,
) -> None:
    if not detections:
        return

    prefix = f"[Frame {frame_idx}] " if frame_idx >= 0 else ""
    print(f"{prefix}=== Manhole Detections ({len(detections)}) ===")

    for i, det in enumerate(detections, 1):
        bbox = det["bbox"]
        print(
            f"  {i}. "
            f"ID={det.get('track_id', '-'):<4} "
            f"{det.get('label', 'Manhole'):15s} "
            f"status={det.get('status', '-'):6s} "
            f"conf={det.get('confidence', 0):.3f} "
            f"bbox=[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
        )
    print()