## DeepLabV3

[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) - DeepLabV3

Implementation of **DeepLabV3** using PyTorch

### Available architectures 
| DeepLabV3               | Backbone          |mean IoU             |
| ----------------------- | ----------------- |---------------------|
| deeplabv3_resnet50      | resnet50          | |
| deeplabv3_resnet101     | resnet101         |76.14(after 30 epochs)                  |


### Dataset
```
├── deeplabv3 
    ├── annotations
        ├── instances_train.json
        ├── instances_val.json
        ├── instances_test.json
    ├── masks
        ├── train
            ├── xxx1.png(png only)
        ├── test
            ├── yyy1.png(png only)
        ├── val
            ├── zzz1.png(png only)    
    ├── train
        ├── xxx1.png or jpg ...
        ├── xxx2.png or jpg ...
    ├── val
        ├── yyy1.png or jpg ...
        ├── yyy2.png orjpg/...
    ├── test
        ├── zzz1.png or jpg ...
        ├── zzz2.png or jpg ...
```

### Train
**Note**: Modify these arguments according to your data and model in `Main.py`
```
python Main.py --train --epochs 30  
```

### Val and Predict
```
python Main.py
```



