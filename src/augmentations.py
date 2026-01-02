import albumentations as A
import numpy as np


def apply_augmentation(image: np.ndarray, degradation_type: str, severity: int) -> np.ndarray:
    """
    Apply augmentation to a single image.
    """
    
    if severity < 1 or severity > 5:
        raise ValueError("Severity must be between 1 and 5")
    
    if degradation_type == "brightness":
        limits = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}
        transform = A.RandomBrightnessContrast(
            brightness_limit=limits[severity],
            contrast_limit=0,
            p=1.0
        )
        result = transform(image=image)["image"]
        
    elif degradation_type == "noise":
        std_values = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
        std = std_values[severity]
        noise = np.random.normal(0, std, image.shape)
        result = image.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)
        
    elif degradation_type == "blur":
        blur_sizes = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11}
        transform = A.GaussianBlur(
            blur_limit=(blur_sizes[severity], blur_sizes[severity]),
            p=1.0
        )
        result = transform(image=image)["image"]
        
    else:
        raise ValueError(f"Unknown degradation: {degradation_type}")
    
    return result