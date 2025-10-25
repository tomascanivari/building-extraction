import gc

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    SegformerForSemanticSegmentation
)
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor

dataset = load_dataset("tomascanivari/building_extraction", split="train")

for idx in tqdm(range(len(dataset))):
    # load annotation lazily
    ann = dataset[idx]["annotation"]

    # convert to numpy
    mask = np.array(ann, dtype=np.int16)[:, :, 0]  # red channel
    mask -= 1
    mask[mask == -1] = 255
    dataset[idx]["annotation"] = mask.astype(np.uint8)

    # optionally, do something with the mask immediately
    # e.g., save to disk
    # Image.fromarray(mask).save(f"/content/masks/{idx}.png")

# Optional: free memory
gc.collect()

# Test Split Annotation is Place-Holder
print(dataset[0])