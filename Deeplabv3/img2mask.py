from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
# Paths to annotations and images
annotation_path = 'deeplabv3/annotations/instances_valid.json'
coco = COCO(annotation_path)
mask_dir = 'deeplabv3/masks/'
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)
# Category mappings (you can adjust these as per your dataset)
category_mapping = {1: 1, 2: 2, 3: 3}  # {COCO category ID: your category ID}

for img_id in coco.getImgIds():
    img = coco.imgs[img_id]
    
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id'], iscrowd=None))
    img_info = coco.loadImgs(img_id)[0]
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    
    if not anns:
        print("没有注释的图片 跳过,它的info是",img_info)
        continue
    else:
        mask = coco.annToMask(anns[0])
        for ann in anns:
            category_id = ann['category_id']
            if category_id in category_mapping:
                mask_instance = coco.annToMask(ann)
                mask[mask_instance == 1] = category_mapping[category_id]
        
        

    # Save mask
    mask_img = Image.fromarray(mask)
    mask_img.save(os.path.join(mask_dir, 'val',f"{img_info['file_name'].split('.')[0]}.png"))