import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from augmentations import apply_augmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#load a sample image
img_path = next(Path("data/mvtec/screw/test/scratch_head").glob("*png"))
img = np.array(Image.open(img_path))

#Visualise all severities for each degradation
fig, axes = plt.subplots(3, 6, figsize=(15,8))

for row, deg_type in enumerate(["brightness", "noise", "blur"]):
    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_title("Original")
    
    for severity in range(1, 6):
        aug_img = apply_augmentation(img, deg_type, severity)
        axes[row, severity].imshow(aug_img, cmap='gray')
        axes[row, severity].set_title(f"{deg_type}\nsev={severity}")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig("results/augmentation_examples.png", dpi=150)
plt.show()