from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

# Paths to annotations and images
annotation_path = 'deeplabv3/annotations/instances_test.json'
coco = COCO(annotation_path)
mask_dir = 'deeplabv3/masks/test/'
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# Loop through each image in the annotations
for img_id in coco.getImgIds():
    img = coco.imgs[img_id]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id'], iscrowd=None))
    img_info = coco.loadImgs(img_id)[0]
    
    # Initialize an empty mask
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    
    if not anns:
        print("No annotations found, producing an all-zero mask. Info:", img_info)
        image_path = os.path.join('deeplabv3/test/', img_info['file_name'])
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        continue
    else:
        # Create the mask based on annotations
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])

        # Check unique classes in the mask
        unique_classes = np.unique(mask)
        print(f"Unique classes in mask for image {img_id}: {unique_classes}")
        
        # Skip saving if unexpected classes are present
        if not np.array_equal(unique_classes, np.array([0, 1, 2, 3])):
            print(f"Unexpected class found in mask for image {img_id}. Deleting corresponding image.")
            image_path = os.path.join('deeplabv3/test/', img_info['file_name'])
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted image: {image_path}")
            continue  # Skip saving this mask if it has unexpected classes

    # Save the mask if it only contains valid classes
    mask_img = Image.fromarray(mask)
    mask_img.save(os.path.join(mask_dir, f"{img_info['file_name'].split('.')[0]}.png"))
    print(f"Saved mask for image {img_id}")