import pyrealsense2 as rs
import numpy as np
import cv2
import math
import camera
import utils


#create an camera instance
cam = camera.Camera()

# We will be removing the background of objects more than clipping_distance_in_meters meters away
clipping_distance_in_meters = 10 #1 meter
clipping_distance = clipping_distance_in_meters / cam.depth_scale

# rs.align allows us to perform alignment of depth frames to others frames The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.depth
align = rs.align(align_to)


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
        img = utils.remove_bg(depth_image, color_image, clipping_distance, grey_color=137)

        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == 13:
            break

    cv2.imshow('image', img)
    param = (img, depth_frame, depth_intrin)
    cv2.setMouseCallback('image', utils.click_event_semi, param=param) 
    key = cv2.waitKey(0)
        # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
finally:
    cam.pipeline.stop()
