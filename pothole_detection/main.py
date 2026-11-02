
import cv2
import sys

from config import settings
from detector import PotholeDetector
from boundix_box import (
    draw_bounding_boxes,
    print_detections
)


def main():

    print("=" * 60)
    print("        POTHOLE DETECTION SYSTEM")
    print("          YOLO + BYTETRACK")
    print("=" * 60)

    print(f"Model       : {settings.MODEL_PATH}")
    print(f"Video       : {settings.VIDEO_SOURCE}")
    print(f"Confidence  : {settings.CONFIDENCE_THRESHOLD}")
    print(f"Show Window : {settings.SHOW_WINDOW}")
    print(f"Print Dets  : {settings.PRINT_DETECTIONS}")

    print("-" * 60)

    # --------------------------------------------------
    # Initialize detector
    # --------------------------------------------------

    try:

        detector = PotholeDetector()

    except Exception as e:

        print(
            f"[FATAL] Failed to load pothole model: {e}"
        )

        sys.exit(1)

    # --------------------------------------------------
    # Open video
    # --------------------------------------------------

    cap = cv2.VideoCapture(
        settings.VIDEO_SOURCE
    )

    if not cap.isOpened():

        print(
            f"[FATAL] Cannot open video source: "
            f"{settings.VIDEO_SOURCE}"
        )

        sys.exit(1)

    # --------------------------------------------------
    # Video information
    # --------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(f"[INFO] Video FPS  : {fps:.2f}")
    print(f"[INFO] Video Size : {width} x {height}")

    # --------------------------------------------------
    # Video writer
    # --------------------------------------------------

    writer = None

    if settings.SAVE_OUTPUT:

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            settings.OUTPUT_PATH,
            fourcc,
            fps,
            (width, height)
        )

        print(
            f"[OK] Saving output to: "
            f"{settings.OUTPUT_PATH}"
        )

    # --------------------------------------------------
    # Tracking statistics
    # --------------------------------------------------

    frame_idx = 0

    # Stores all unique ByteTrack IDs
    unique_pothole_ids = set()

    # --------------------------------------------------
    # Start detection
    # --------------------------------------------------

    print()
    print(
        "[INFO] Starting pothole detection..."
    )

    print(
        "[INFO] YOLO + ByteTrack active."
    )

    print(
        "[INFO] Press 'q' or ESC to exit."
    )

    print()

    try:

        while True:

            # --------------------------------------------------
            # Read frame
            # --------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                print(
                    "[INFO] End of video / stream."
                )

                break

            frame_idx += 1

            # --------------------------------------------------
            # YOLO + ByteTrack detection
            # --------------------------------------------------

            detections = detector.detect(frame)

            # --------------------------------------------------
            # Store unique ByteTrack IDs
            # --------------------------------------------------

            for det in detections:

                track_id = det.get("track_id")

                if track_id is not None:

                    unique_pothole_ids.add(
                        track_id
                    )

            # --------------------------------------------------
            # Console output
            # --------------------------------------------------

            if (
                settings.PRINT_DETECTIONS
                and detections
            ):

                print_detections(
                    detections,
                    frame_idx
                )

            # --------------------------------------------------
            # Draw bounding boxes + IDs
            # --------------------------------------------------

            annotated = draw_bounding_boxes(
                frame,
                detections
            )

            # --------------------------------------------------
            # Current frame information
            # --------------------------------------------------

            current_potholes = len(
                detections
            )

            total_unique_potholes = len(
                unique_pothole_ids
            )

            # --------------------------------------------------
            # Status text 1
            # --------------------------------------------------

            status = (
                f"Frame: {frame_idx} | "
                f"Current Potholes: "
                f"{current_potholes}"
            )

            cv2.putText(
                annotated,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # --------------------------------------------------
            # Status text 2
            # --------------------------------------------------

            total_status = (
                f"Unique Potholes Detected: "
                f"{total_unique_potholes}"
            )

            cv2.putText(
                annotated,
                total_status,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # --------------------------------------------------
            # Save annotated video
            # --------------------------------------------------

            if writer is not None:

                writer.write(
                    annotated
                )

            # --------------------------------------------------
            # Display
            # --------------------------------------------------

            if settings.SHOW_WINDOW:

                display_frame = cv2.resize(
                    annotated,
                    (1500 , 844 ) 
                )

                cv2.imshow(
                    "Pothole Detection - YOLO + ByteTrack",
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:

                    print(
                        "[INFO] Quit requested."
                    )

                    break

    except KeyboardInterrupt:

        print(
            "\n[INFO] Interrupted by user."
        )

    finally:

        # --------------------------------------------------
        # Release resources
        # --------------------------------------------------

        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        # --------------------------------------------------
        # Final statistics
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("              FINAL STATISTICS")
        print("=" * 60)

        print(
            f"Processed Frames       : "
            f"{frame_idx}"
        )

        print(
            f"Unique Potholes        : "
            f"{len(unique_pothole_ids)}"
        )

        print(
            f"Unique Track IDs       : "
            f"{sorted(unique_pothole_ids)}"
        )

        print("=" * 60)
        print("Pothole detection completed.")
        print("=" * 60)


if __name__ == "__main__":
    main()

