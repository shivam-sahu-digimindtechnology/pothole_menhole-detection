from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Manhole YOLO model (open / closed)
    MODEL_PATH: str = r"F:\pothole\manhole\hole.pt"

    # Input video
    VIDEO_SOURCE: str | int = r"F:\pothole\manhole\data\manhole_vdo.mp4"

    # Detection confidence
    CONFIDENCE_THRESHOLD: float = 0.5

    # Display / output options
    SHOW_WINDOW: bool = True
    PRINT_DETECTIONS: bool = True

    # Save annotated video
    SAVE_OUTPUT: bool = True
    OUTPUT_PATH: str = r"F:\pothole\manhole\manhole_output.mp4"


settings = Settings()