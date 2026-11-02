import cv2
import sys
import os

from config import settings
from detector import ManholeDetector
from bounding_box import (
    draw_bounding_boxes,
    draw_stats_overlay,
    print_detections,
)


def main():
    print("=" * 60)
    print("       MANHOLE DETECTION SYSTEM")
    print("     Open Manhole / Manhole + ByteTrack")
    print("=" * 60)

    print(f"Model       : {settings.MODEL_PATH}")
    print(f"Video       : {settings.VIDEO_SOURCE}")
    print(f"Confidence  : {settings.CONFIDENCE_THRESHOLD}")
    print(f"Show Window : {settings.SHOW_WINDOW}")
    print(f"Print Dets  : {settings.PRINT_DETECTIONS}")
    print("-" * 60)

    # Load model
    try:
        detector = ManholeDetector()
    except Exception as e:
        print(f"[FATAL] Failed to load manhole model: {e}")
        sys.exit(1)

    # Open input video
    cap = cv2.VideoCapture(settings.VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"[FATAL] Cannot open video source: {settings.VIDEO_SOURCE}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video FPS  : {fps:.2f}")
    print(f"[INFO] Video Size : {width} x {height}")

    writer = None
    output_path = settings.OUTPUT_PATH if settings.SAVE_OUTPUT else None

    try:
        # Create output directory and initialize video writer
        if settings.SAVE_OUTPUT:
            output_path = os.path.abspath(settings.OUTPUT_PATH)
            output_dir = os.path.dirname(output_path)

            os.makedirs(output_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height),
            )

            if not writer.isOpened():
                writer.release()
                writer = None
                raise RuntimeError(
                    f"Could not initialize output video writer: {output_path}"
                )

            print(f"[OK] Output directory: {output_dir}")
            print(f"[OK] Saving output to: {output_path}")

        frame_idx = 0

        # Unique tracking IDs
        unique_all_ids = set()
        unique_open_ids = set()
        unique_closed_ids = set()

        print("\n[INFO] Starting manhole detection...")
        print("[INFO] YOLO + ByteTrack active.")
        print("[INFO] Press 'q' or ESC to exit.\n")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[INFO] End of video / stream.")
                break

            frame_idx += 1

            # Run detection and tracking
            detections = detector.detect(frame)

            current_open = 0
            current_closed = 0

            for det in detections:
                track_id = det.get("track_id")
                status = str(det.get("status", "")).lower()

                if track_id is None:
                    continue

                unique_all_ids.add(track_id)

                if status == "open":
                    unique_open_ids.add(track_id)
                    current_open += 1
                elif status == "closed":
                    unique_closed_ids.add(track_id)
                    current_closed += 1

            # Print detections
            if settings.PRINT_DETECTIONS and detections:
                print_detections(detections, frame_idx)
                print(
                    f"  --> Unique Total: {len(unique_all_ids)} | "
                    f"Open: {len(unique_open_ids)} | "
                    f"Closed: {len(unique_closed_ids)}\n"
                )

            # Draw detection boxes and labels
            annotated = draw_bounding_boxes(frame, detections)

            # Draw statistics overlay
            annotated = draw_stats_overlay(
                annotated,
                total=len(unique_all_ids),
                open_count=len(unique_open_ids),
                closed_count=len(unique_closed_ids),
                frame_idx=frame_idx,
                current_open=current_open,
                current_closed=current_closed,
            )

            # Save every processed frame
            if writer is not None:
                writer.write(annotated)

            # Display output
            if settings.SHOW_WINDOW:
                display_frame = cv2.resize(
                    annotated,
                    (1500, 844),
                )

                cv2.imshow(
                    "Manhole Detection - Open / Closed",
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    print("[INFO] Quit requested.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        raise

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("              FINAL STATISTICS")
        print("=" * 60)
        print(f"Processed Frames       : {frame_idx if 'frame_idx' in locals() else 0}")
        print(f"Unique Manholes Total  : {len(unique_all_ids) if 'unique_all_ids' in locals() else 0}")
        print(f"Unique Open            : {len(unique_open_ids) if 'unique_open_ids' in locals() else 0}")
        print(f"Unique Closed          : {len(unique_closed_ids) if 'unique_closed_ids' in locals() else 0}")

        if 'unique_open_ids' in locals():
            print(f"Open Track IDs         : {sorted(unique_open_ids)}")
            print(f"Closed Track IDs       : {sorted(unique_closed_ids)}")

        if output_path and os.path.isfile(output_path):
            print(f"\n[OK] Output video saved: {output_path}")
            print(f"[OK] File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        elif settings.SAVE_OUTPUT:
            print("[WARNING] Output video was not created.")

        print("=" * 60)
        print("Manhole detection completed.")
        print("=" * 60)


if __name__ == "__main__":
    main()
