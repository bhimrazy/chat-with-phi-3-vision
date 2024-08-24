from src.config import PHI_VISION_MODELS


def is_valid_model(model_name: str) -> bool:
    return model_name in PHI_VISION_MODELS
