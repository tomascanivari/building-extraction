import json
import os
from PIL import Image

# --- USER SETTINGS ---
images_dir = "image"             # Folder with your original images
output_dir = "image_512"         # Folder to save resized images
json_file = "val.json"    # Original COCO JSON
output_json = "val_512.json"
new_size = (512, 512)             # Width, Height

os.makedirs(output_dir, exist_ok=True)

# --- LOAD COCO JSON ---
with open(json_file, 'r') as f:
    coco = json.load(f)

# --- MAP IMAGE ID TO FILE PATH ---
id_to_image = {img['id']: img for img in coco['images']}

# --- RESIZE IMAGES AND UPDATE ANNOTATIONS ---
for img_info in coco['images']:
    file_name = img_info['file_name'].split("/")[1]
    img_path = os.path.join(images_dir, file_name)
    img = Image.open(img_path)
    old_width, old_height = img.size

    # Resize image
    img_resized = img.resize(new_size)
    img_resized.save(os.path.join(output_dir, file_name))

    # Compute scale factors
    scale_x = new_size[0] / old_width
    scale_y = new_size[1] / old_height

    # Update image info in JSON
    img_info['width'], img_info['height'] = new_size

    # Update image info in JSON
    img_info['file_name'] = os.path.join(output_dir, file_name)


    # Update annotations for this image
    for ann in coco['annotations']:
        if ann['image_id'] != img_info['id']:
            continue

        # Update bbox: [x_min, y_min, width, height]
        x, y, w, h = ann['bbox']
        ann['bbox'] = [
            x * scale_x,
            y * scale_y,
            w * scale_x,
            h * scale_y
        ]

        # Update segmentation polygons
        for seg in ann['segmentation']:
            for i in range(0, len(seg), 2):
                seg[i] *= scale_x       # x
                seg[i+1] *= scale_y     # y

# --- SAVE UPDATED JSON ---
with open(output_json, 'w') as f:
    json.dump(coco, f)

print("All images resized and annotations updated!")