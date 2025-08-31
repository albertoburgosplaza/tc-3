import os
from typing import Dict, Union

# COCO class mapping for object detection - keeping only person and car
CLASSES_KEEP: Dict[str, int] = {
    "person": 0,
    "car": 2
}

# Load environment variables with default values
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/yolov8n.pt")

# Detection parameters with validation
def _validate_threshold(value: Union[str, float], name: str, default: float) -> float:
    """Validate threshold values are in range [0.0, 1.0]"""
    try:
        val = float(value) if isinstance(value, str) else value
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {val}")
        return val
    except (ValueError, TypeError):
        print(f"Warning: Invalid {name} value '{value}', using default {default}")
        return default

# Confidence threshold for detections
DEFAULT_CONF: float = _validate_threshold(
    os.getenv("CONF", "0.25"), "CONF", 0.25
)

# IoU threshold for Non-Maximum Suppression
DEFAULT_IOU: float = _validate_threshold(
    os.getenv("IOU", "0.45"), "IOU", 0.45
)

# Maximum number of detections
DEFAULT_MAX_DETECTIONS: int = 300