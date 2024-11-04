## Yolo11seg

Implementation of **Yolo11seg** using PyTorch

### Available architectures 
                  |
YOLO11seg     
### Dataset
```
├── YOLO-SEG 
    ├── annotations
        ├── train.json
        ├── val.json
        ├── itest.json
    ├── dataset
        ├── images
            ├── train
                ├── xxx1.jpg JPEG
            ├── test
                ├── yyy1.jpg JPEG
            ├── val
                ├── zzz1.jpg JPEG
        ├── labels
            ├── train
                ├── xxx1.txt
            ├── test
                ├── yyy1.txt
            ├── val
                ├── zzz1.txt     
    
```
### Preprocess
**Note** 
json2yolo.py Converts COCO JSON format to YOLO label format
split_images.py  Divide the dataset into train,test,val
### Train
**Note Before you run the Train.py** 
```
1.Modify the path in the yoloseg.yaml
2.python Train.py
3.Locate the newly generated best.pt file in the runs/segment folder, e.g. "runs/segment/train8/best.pt".
Change the path to the loaded model in the Val section on YoloSeg.ipynb

OR

1. Running directly in YoloSeg.ipynb
```
### Val and Predict
```
1. Running directly in YoloSeg.ipynb
2. The validated IOU metrics and predicted mask images are saved in the "/out" directory
```