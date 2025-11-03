import os
import json
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

# Setup logger
setup_logger()

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

# ====== Dataset Registration ======
def register_custom_dataset():
    classes = ["fish"]  # Replace with your actual class names

    DatasetCatalog.register("custom_train", lambda: get_custom_dataset_dicts("/scratch/daff/wangj2/fish_size_maskrcnn/fish_coco_dataset/train_images/", "/scratch/daff/wangj2/fish_size_maskrcnn/fish_coco_dataset/annotations/train_annotations.json"))
    MetadataCatalog.get("custom_train").set(thing_classes=classes)
    DatasetCatalog.register("custom_val", lambda: get_custom_dataset_dicts("/scratch/daff/wangj2/fish_size_maskrcnn/fish_coco_dataset/test_images/", "/scratch/daff/wangj2/fish_size_maskrcnn/fish_coco_dataset/annotations/test_annotations.json"))
    MetadataCatalog.get("custom_val").set(thing_classes=classes)

# ====== Main Training Code ======
def main():
    # Register datasets
    register_custom_dataset()

    # Setup config
    cfg = get_cfg()

    # Load local config file (instead of downloading from model zoo)
    cfg.merge_from_file("/scratch/daff/wangj2/fish_size_maskrcnn/mask_rcnn_R_50_FPN_3x.yaml")  # <== UPDATE path

    # Load local pretrained weights (also downloaded manually)
    cfg.MODEL.WEIGHTS = "/scratch/daff/wangj2/fish_size_maskrcnn/model_final_f10217.pkl"    # <== UPDATE path

    # Dataset names
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",)

    # Set number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Update based on your dataset

    # Solver settings
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Output dir
    cfg.OUTPUT_DIR = "/scratch/daff/wangj2/fish_size_maskrcnn/output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
