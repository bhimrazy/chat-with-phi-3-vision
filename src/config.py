IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v"]
MAX_NUM_FRAMES = 64

SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a helpful assistant.",
}

PHI_VISION_MODELS = {
    "phi-3-vision": "microsoft/Phi-3-vision-128k-instruct",
    "phi-3.5-vision": "microsoft/Phi-3.5-vision-instruct",
}

DEFAULT_MODEL = PHI_VISION_MODELS["phi-3.5-vision"]
