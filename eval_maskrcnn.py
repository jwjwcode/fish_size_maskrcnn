from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
import os
import torch
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

def register_custom_dataset():
    classes = ["fish"]  # Replace with your actual class names

    DatasetCatalog.register("custom_train", lambda: get_custom_dataset_dicts("/scratch/daff/wangj2/fish_size_maskrcnn/fish_coco_dataset/train_images/", "/scratch/daff/wangj2/fish_size_maskrcnn/fish_coco_dataset/annotations/train_annotations.json"))
    MetadataCatalog.get("custom_train").set(thing_classes=classes)
    DatasetCatalog.register("custom_val", lambda: get_custom_dataset_dicts("/scratch/daff/wangj2/fish_size_maskrcnn/fish_coco_dataset/test_images/", "/scratch/daff/wangj2/fish_size_maskrcnn/fish_coco_dataset/annotations/test_annotations.json"))
    MetadataCatalog.get("custom_val").set(thing_classes=classes)

# ====== Dataset Loader from COCO-style JSON ======
def get_custom_dataset_dicts(img_dir, annotation_file):
    with open(annotation_file) as f:
        coco_data = json.load(f)

    dataset_dicts = []
    for img in coco_data["images"]:
        record = {
            "file_name": os.path.join(img_dir, img["file_name"]),
            "image_id": img["id"],
            "height": img["height"],
            "width": img["width"],
            "annotations": [],
        }

        for ann in coco_data["annotations"]:
            if ann["image_id"] == img["id"]:
                obj = {
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": ann["segmentation"],
                    "category_id": ann["category_id"] - 1,
                    "iscrowd": ann.get("iscrowd", 0),
                }
                record["annotations"].append(obj)

        dataset_dicts.append(record)

    return dataset_dicts


register_custom_dataset()
# Step 1: Set up your configuration
cfg = get_cfg()
cfg.merge_from_file("/scratch/daff/wangj2/fish_size_maskrcnn/mask_rcnn_R_50_FPN_3x.yaml")  # Point to your trained config file
cfg.MODEL.WEIGHTS = "/scratch/daff/wangj2/fish_size_maskrcnn/output/model_final.pth"  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Update based on your dataset
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for predictions (you can adjust this)
cfg.DATASETS.TEST = ("custom_val",)  # The name of your test dataset

# Step 2: Create the predictor to run inference
predictor = DefaultPredictor(cfg)

# Step 3: Set up the evaluator and test data loader
evaluator = COCOEvaluator("custom_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "custom_val")

# Step 4: Run the evaluation and print the metrics
metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
print(metrics)
