## DeepLabV3

[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) - DeepLabV3

Implementation of **DeepLabV3** using PyTorch

### Available architectures 
| DeepLabV3               | Backbone          |mean IoU             |
| ----------------------- | ----------------- |---------------------|
| deeplabv3_resnet50      | resnet50          |58.6(after 5 epochs) |
| deeplabv3_resnet101     | resnet101         |-                    |
| deeplabv3_mobilenetv3   | mobilenetv3_large |-                    |

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
**Note**: Modify these arguments according to your data and model in `CustomDataset.py`
```
num_epochs =??

python CustomDataset.py
     
```

**Distributed Data Parallel:** 

1. deeplabv3_resnet50:

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --lr 0.02 --dataset coco -b 8 --model deeplabv3_resnet50

```

1. deeplabv3_resnet101:


```
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --lr 0.02 --dataset coco -b 8 --model deeplabv3_resnet101

```

1. deeplabv3_mobilenet_v3_large:

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --lr 0.02 --dataset coco -b 8 --model deeplabv3_mobilenet_v3_large



