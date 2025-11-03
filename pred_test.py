import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from centerLineExtract import get_centerline

# --- Config ---
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1   # only 1 class (fish)
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

# --- Read image ---
img = cv2.imread("test.png")

# --- Run inference ---
outputs = predictor(img)

get_centerline(outputs, img)

# --- Create metadata for your class ---
fish_metadata = MetadataCatalog.get("fish_dataset")
fish_metadata.set(thing_classes=["fish"])

# --- Visualization ---
v = Visualizer(img[:, :, ::-1], metadata=fish_metadata, scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Convert back to BGR for OpenCV
result_img = out.get_image()[:, :, ::-1]

# --- Show and save ---
cv2.imshow("Detections", result_img)
cv2.imwrite("result.png", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


 

