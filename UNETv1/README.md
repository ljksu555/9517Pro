## Turtles Dataset

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
```

## Usage

1. **Modify Paths in Jupyter Notebook**: Open the Jupyter Notebook you intend to use with this dataset. Update the dataset paths (e.g., `annotations.json`, etc) in the notebook to match your local directory structure.

2. **Run the Notebook**: After updating the paths, run the notebook cells to load and process the dataset for training or evaluation tasks.

3. **Training and Evaluation**: Use the dataset to train object detection models like YOLO or any compatible model. Ensure `data.yaml` is properly configured to reflect the dataset structure and class information.
