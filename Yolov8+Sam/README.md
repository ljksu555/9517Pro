
# YOLOv8 and SAM 

This repository demonstrates how to train a YOLOv8 model for object detection and integrate it with a SAM model for segmentation tasks using the turtles dataset. Follow the steps below to set up the environment and execute the code.

---

## Directory Structure

Ensure that your `base_path` directory is structured as follows:

```plaintext
├── base_path
|   ├── turtles-data
|   |    ├── data
|   |    │   ├── annotations
|   |    │   ├── images
|   |    │   ├── metadata_splits.csv
|   |    │   ├── metadata.csv
|   |    │   ├── annotations.json
```

---

## Steps to Set Up

### 1. Define `base_path`

In the yolov8+sam.ipynb, **modify the 6th line** to specify your `base_path`. For example:

```python
base_path = "/root/autodl-fs"
```

### 2. Create `data.yaml`

Navigate to `f'{base_path}/archive/turtles-data'` and create a new file named `data.yaml`. The file should contain the following content, with `base_path` replaced by your actual directory path:

#### Example `data.yaml`

```yaml
# Path to the training dataset
train: /root/autodl-fs/archive/turtles-data/images/train

# Path to the validation dataset
val: /root/autodl-fs/archive/turtles-data/images/test

# Number of classes (assuming there are 3 classes; modify to actual class count)
nc: 3

# Class names
names: ['shell', 'fin', 'head']
```

---

### 3. Download the SAM Model Checkpoint

The code requires a specific checkpoint file for the SAM model. Download the `sam_vit_h_4b8939.pth` file and place it in your `base_path`.

- **Download Link**: [sam_vit_h_4b8939.pth](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth)

After downloading, your directory should look like this:

```plaintext
├── base_path
|   ├── sam_vit_h_4b8939.pth
|   ├── turtles-data
|   |    ├── data
|   |    │   ├── annotations
|   |    │   ├── images
|   |    │   ├── metadata_splits.csv
|   |    │   ├── metadata.csv
|   |    │   ├── annotations.json
```

---

## Steps to Run the Code

### 1. Install Dependencies

Ensure all required Python packages are installed. Run:

```bash
pip install torch numpy pandas matplotlib seaborn Pillow scikit-learn pycocotools ultralytics segment-anything

```

### 2. Run the code in yolov8+sam.ipynb


