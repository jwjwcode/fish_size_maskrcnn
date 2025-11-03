import pyrealsense2 as rs
import numpy as np
import cv2
import math
import camera
import utils
from ultralytics import YOLO

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_



SEG_MODEL = 'maskrcnn' # model can be yolo or maskrcnn

#create an camera instance
cam = camera.Camera()

# We will be removing the background of objects more than clipping_distance_in_meters meters away
clipping_distance_in_meters = 5 #1 meter
clipping_distance = clipping_distance_in_meters / cam.depth_scale

# rs.align allows us to perform alignment of depth frames to others frames The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.depth
align = rs.align(align_to)

if SEG_MODEL == 'YOLO':
# Load a pretrained YOLO11n model
    model = YOLO("best_xlarge_640.engine", task='segment')
    _ = model(np.ones((640,640,3)))
    
elif SEG_MODEL == 'maskrcnn':
    # --- Config ---
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1   # only 1 class (fish)
    cfg.MODEL.DEVICE = "cuda"
    model = DefaultPredictor(cfg)
    
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = cam.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image


        # Align the color frame to depth frame
        depth_frame, aligned_color_frame = utils.align_frame(align, frames)
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
 
        # Validate that both frames are valid
        if not depth_frame or not aligned_color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())
        # Remove background - Set pixels further than clipping_distance to grey
        bg_removed_img = utils.remove_bg(depth_image, color_image, clipping_distance, grey_color=137)

        cv2.imshow('image', bg_removed_img)
        key = cv2.waitKey(1)
        if key == 13:
            break

    cv2.imshow('image', bg_removed_img)
    
    if SEG_MODEL == 'YOLO':
        results = model(color_image, save=False, imgsz=640, conf=0.5) 
        result = results[0]
        if result is None:
            print('no fish detected')
        else:        
            masks = result.masks.data.cpu().numpy()  
    elif SEG_MODEL == 'maskrcnn':
        print('using maskrcnn')
        results = model(color_image) 
        result = results["instances"]
        if result is None:
            print('no fish detected')
        else:
            masks = result.pred_masks.cpu().numpy()
    param_auto = (bg_removed_img, depth_frame, depth_intrin, masks, color_image, cam)
    cv2.setMouseCallback('image', utils.click_event_auto, param=param_auto) 
    key = cv2.waitKey(0)
        # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
finally:
    cam.pipeline.stop()
    print('camera closed')
