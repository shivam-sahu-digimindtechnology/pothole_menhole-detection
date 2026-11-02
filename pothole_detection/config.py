
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Custom trained pothole YOLO model
    # MODEL_PATH: str = r"F:\pothole\pothole_detection\potholeD.pt"
    MODEL_PATH: str = r"F:\pothole\manhole\hole.pt"

    # Input video
    VIDEO_SOURCE: str | int = r"F:\pothole\pothole_detection\pothole.mp4"
    VIDEO_SOURCE: str | int = r"F:\pothole\manhole\manhole_vdo.mp4"

    # Detection confidence
    CONFIDENCE_THRESHOLD: float = 0.5

    # Display / output options
    SHOW_WINDOW: bool = True
    PRINT_DETECTIONS: bool = True

    # Save annotated video
    SAVE_OUTPUT: bool = False
    OUTPUT_PATH: str = r"F:\pothole\pothole_detection\pothole_output.mp4"


settings = Settings()
