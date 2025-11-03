import pyrealsense2 as rs
import numpy as np
import cv2
import math

class Camera():
    '''class for the connected realsense 3D camera'''
    def __init__(self):
        '''initialize the camera'''
        #Create a pipeline
        self.pipeline = rs.pipeline()
      # Create a config and configure the pipeline to stream different resolutions of color and depth streams
        self.config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        self.check_rgb_exist()
        self.check_rgb_exist()
        self.enable_and_start_stream()
        self.get_depthscale()
        
    def check_rgb_exist(self):
        '''check if there is rgb camera, if not exit'''
        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
            
    def enable_and_start_stream(self):
        '''enable camera stream of depth and color'''
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
    def get_depthscale(self):
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        print('depth scale is:',self.depth_scale)



