## Turtles Dataset

This dataset is designed for training and evaluating object detection models on turtle images. It includes annotations and images divided into training and testing sets, with configuration details provided in `data.yaml`.

```plaintext
turtles-data
├── data
│   ├── annotations
│   │   ├── annotations_train.json
│   │   ├── annotations_test.json
│   │   ├── annotations_valid.json
│   ├── metadata_splits.csv
│   ├── metadata.csv
│   ├── annotations.json
├── images
│   ├── train
│   │   ├── xxx1.png or .jpg
│   │   ├── xxx2.png or .jpg
│   │   └── ...
│   ├── test
│   │   ├── yyy1.png or .jpg
│   │   ├── yyy2.png or .jpg
│   │   └── ...
├── labels
│   ├── train
│   │   ├── xxx1.txt
│   │   ├── xxx2.txt
│   │   └── ...
│   ├── test
│   │   ├── yyy1.txt
│   │   ├── yyy2.txt
│   │   └── ...
├── runs
│   └── ...
└── data.yaml
```

### data.yaml Configuration

- `train`: Path to the training images.
- `val`: Path to the validation images.
- `nc`: Number of classes (set to 3 in this dataset, but modify as needed).
- `names`: List of class names, which in this dataset are `shell`, `fin`, and `head`.

### Example `data.yaml`

```yaml
# Path to the training dataset
train: D:/archive/turtles-data/images/train

# Path to the validation dataset
val: D:/archive/turtles-data/images/test

# Number of classes (assuming there are 3 classes; modify to actual class count)
nc: 3

# Class names
names: ['shell', 'fin', 'head']

```
## Usage

1. **Modify Paths in Jupyter Notebook**: Open the Jupyter Notebook you intend to use with this dataset. Update the dataset paths (e.g., `data.yaml`, `images/train`, `images/test`, and `annotations.json`, etc) in the notebook to match your local directory structure.

2. **Run the Notebook**: After updating the paths, run the notebook cells to load and process the dataset for training or evaluation tasks.

3. **Training and Evaluation**: Use the dataset to train object detection models like YOLO or any compatible model. Ensure `data.yaml` is properly configured to reflect the dataset structure and class information.
