"""Severity-based camera degradations (brightness shift, Gaussian sensor noise, focus blur).

All randomness comes from an explicit numpy Generator, so results are reproducible
given a seed. `apply_augmentation_logged` also returns the sampled parameter.
"""
import cv2
import numpy as np

BRIGHTNESS_LIMITS = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}  # max |shift| as fraction of 255
NOISE_STD = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}                 # sigma in 8-bit grey levels
BLUR_KERNEL = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11}                    # Gaussian kernel size (sigma from size)


def apply_augmentation_logged(image: np.ndarray, degradation_type: str, severity: int,
                              rng: np.random.Generator | None = None):
    """Apply one degradation to a uint8 HxWxC image. Returns (image, parameter)."""
    if severity < 1 or severity > 5:
        raise ValueError("Severity must be between 1 and 5")
    rng = rng if rng is not None else np.random.default_rng()

    if degradation_type == "brightness":
        lim = BRIGHTNESS_LIMITS[severity]
        shift = float(rng.uniform(-lim, lim))  # same distribution as albumentations RandomBrightnessContrast
        out = np.clip(image.astype(np.float32) + shift * 255.0, 0, 255).astype(np.uint8)
        return out, shift

    if degradation_type == "noise":
        std = NOISE_STD[severity]
        noise = rng.normal(0.0, std, image.shape)
        out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out, float(std)

    if degradation_type == "blur":
        k = BLUR_KERNEL[severity]
        out = cv2.GaussianBlur(image, (k, k), 0)  # deterministic: sigma derived from kernel size
        return out, float(k)

    raise ValueError(f"Unknown degradation: {degradation_type}")


def apply_augmentation(image: np.ndarray, degradation_type: str, severity: int,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Backward-compatible wrapper returning only the image."""
    return apply_augmentation_logged(image, degradation_type, severity, rng)[0]