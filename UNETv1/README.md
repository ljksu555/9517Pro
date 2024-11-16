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

In the UNETv1.ipynb, **modify the 6th line** to specify your `base_path`. For example:

```python
base_path = "/root/autodl-fs"
```

## Steps to Run the Code

### 1. Install Dependencies

Ensure all required Python packages are installed. Run:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn Pillow scikit-learn pycocotools
```

### 2. Run the code in UNETv1.ipynb


