import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter
import pyrealsense2 as rs
from ultralytics import YOLO
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter
import utils


class ColorResults():
    '''mask class to manage 2d prediction results'''
    def __init__(self,masks, color_image):
        '''initialize the results with image and yolo predictions'''
        self.color_image = color_image
        self.masks = masks
        self.skeletons, self.skeletons_points = self.get_skeleton()
        self.lines = self.get_lines()
        self.polys = self.get_polys()
        
    def get_skeleton(self, save=True):
        '''skeletonize the mask to obtain single pixel wide curve'''
        self.skeletons = []
        self.skeletons_points = []
        image_with_mask = self.color_image.copy()
        image_with_skeleton = self.color_image.copy()
        for i in range(self.masks.shape[0]):
            mask = self.masks[i,:,:] 
            if mask.shape[0] == mask.shape[1]:
                mask = mask[80:560,:] # slice the middle of the square part to the original image size
            #mask = utils.process_mask(mask)
            skeleton = self.skeletonize_with_smooth(mask)
            skeleton_255 = (255*skeleton).astype(np.uint8)
            skeleton_points = np.column_stack(np.where(skeleton_255 == 255)) #get the coords of skeleton points
            self.skeletons.append(skeleton)
            self.skeletons_points.append(skeleton_points)
            if save: # choose to save the masks and skeleton.                
                image_with_mask, image_with_skeleton = self.save_skeleton(skeleton, mask, image_with_mask, image_with_skeleton)
        self.skeletons = np.asarray(self.skeletons)
        
        print('ske shape: ', self.skeletons.shape)
        
        return self.skeletons, self.skeletons_points
        
    def get_lines(self, save=True):
        '''get all lines fitted on the skeletons'''
        self.lines = []#each line is represented by the coordinate of starting and end point
        image = self.color_image.copy()
        #loop through all skeletons
        for i in range(self.skeletons.shape[0]):
            skeleton = self.skeletons[i,:,:]
            line = self.detect_line(skeleton, image)
            self.lines.append(line)
            
        return self.lines
        
    def get_polys(self, save=True):
        '''get all lines fitted on the skeletons, return the coords of polys '''
        self.polys = [] # each poly is represented by a list of points
        image = self.color_image.copy()#for save and visuliation
                #loop through all skeletons
        for i in range(self.skeletons.shape[0]):
            skeleton = self.skeletons[i,:,:]
            poly_points = self.detect_poly(skeleton, image)
            self.polys.append(poly_points)
            
        return self.polys
        
        
            
            
    def skeletonize_with_smooth(self, mask, skeleton_settings={'iter_num':1, 'threshold':70,}):
        '''skeleonize the mask and use gaussian filter to smooth the line'''
        skeleton = mask
        #iterate until obtaining a smooth line
        for i in range(skeleton_settings['iter_num']):
            skeleton = skeletonize(skeleton)
            # Convert skeleton to 8-bit format for OpenCV (255 for foreground, 0 for background)
            skeleton = img_as_ubyte(skeleton)
            # Apply Gaussian smoothing to the skeleton
            # The kernel size (sigma) controls the level of smoothing
            smoothed_skeleton = gaussian_filter(skeleton.astype(float), sigma=1)

            # Threshold the smoothed skeleton to convert it back to binary
            _, smoothed_skeleton_binary = cv2.threshold(smoothed_skeleton, skeleton_settings['threshold'], 255, cv2.THRESH_BINARY)
            skeleton = skeletonize(smoothed_skeleton_binary)
            
        #get coords of skeleton
        #skeleton_255 = (255*skeleton).astype(np.uint8)
        #skeleton_points = np.column_stack(np.where(skeleton_255 == 255))
        skeleton, longest_path_mask = utils.process_mask_to_centerline(mask)

        return longest_path_mask
        
    def save_skeleton(self, skeleton, mask, image_with_mask, image_with_skeleton):
        '''save the skeleton overlap on the color image'''
        mask = np.stack((mask,mask,mask), axis=-1)
        skeleton = np.stack([skeleton] * 3, axis=-1)
        image_with_mask = 255*mask + image_with_mask
        image_with_skeleton = 255*skeleton + image_with_skeleton
        
        cv2.imwrite('overlap_xl_v1.png', image_with_mask)
        cv2.imwrite('sk_xl_v1.png', image_with_skeleton) 
        print('save results.....')
    
        return image_with_mask, image_with_skeleton 
            
    def detect_line(self, skeleton, image, save=True):
        '''detect lines and show the end points'''
        
        # Apply linear regression to fit a line

        skeleton = (255*skeleton).astype(np.uint8)
        # Find the coordinates of the white pixels (skeleton points)
        skeleton_points = np.column_stack(np.where(skeleton == 255)) 
       
        # Perform least squares line fitting
        # Use np.polyfit to fit a line to the points
        # polyfit returns the coefficients [m, b] for y = mx + b
        
        x = skeleton_points[:, 1]  # x-coordinates (columns of the image)
        y = skeleton_points[:, 0]  # y-coordinates (rows of the image)
        line_points = []
        if np.max(x) - np.min(x) > np.max(y) - np.min(y): #check the range of x and range of y, the larger one is the x in the fit
        
            m, b = np.polyfit(x, y, 1)
            # Calculate the x range of the skeleton points
            x_min = skeleton_points[:, 1].min()  # Minimum x-coordinate
            x_max = skeleton_points[:, 1].max()  # Maximum x-coordinate

            # Calculate the y-coordinates for the min and max x values
            y_min = int(m * x_min + b)  # y at x_min
            y_max = int(m * x_max + b)  # y at x_max

            # Ensure the y-coordinates are within image bounds
            height, width = image.shape[0], image.shape[1] 
            y_min = max(0, min(height - 1, y_min))
            y_max = max(0, min(height - 1, y_max))    
            x_fit = np.linspace(np.min(x), np.max(x), 100) # number of points should be the same as longer side      
            y_fit = (m * x_fit + b)
            print('horizontal fish line')
            
        else:
 
            m, b = np.polyfit(y, x, 1)
            # Calculate the x range of the skeleton points
            y_min = y.min()  # Minimum x-coordinate
            y_max = y.max()  # Maximum x-coordinate

            # Calculate the y-coordinates for the min and max x values
            x_min = int(m * y_min + b)  # y at x_min
            x_max = int(m * y_max + b)  # y at x_max

            # Ensure the y-coordinates are within image bounds
            height, width = image.shape[0], image.shape[1] 
            y_min = max(0, min(height - 1, y_min))
            y_max = max(0, min(height - 1, y_max))  
            y_fit = np.linspace(np.min(y), np.max(y), 100) # number of points should be the same as longer side          
            x_fit = (m * y_fit + b)       
            print('vertical fish line')
        
        for i in range(len(x_fit)):
            line_points.append((int(y_fit[i]), int(x_fit[i])))
            cv2.circle(image, (int(x_fit[i]), int(y_fit[i])), 1, (0, 0, 255), -1)
        if save:    
            # Draw the fitted line on the original image
            #cv2.line(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red line
            cv2.imwrite('fit_line_v1.png', image)
        
        return np.asarray(line_points)
        
    def detect_poly(self, skeleton, image, save=True):
        '''fit polynomial to the skeleton'''
        
        skeleton = (255*skeleton).astype(np.uint8)
        # Find the coordinates of the white pixels (skeleton points)
        skeleton_points = np.column_stack(np.where(skeleton == 255)) 

        # Step 3: Fit a second-order polynomial (quadratic) to the extracted points
        x = skeleton_points[:, 1]  # x-coordinates (columns of the image)
        y = skeleton_points[:, 0]  # y-coordinates (rows of the image)
        
        if np.max(x) - np.min(x) > np.max(y) - np.min(y): #check the range of x and range of y, the larger one is the x in the fit
    
            # Fit a second-order polynomial (quadratic) to the data
            coefficients = np.polyfit(x, y, 2)
    
            # Generate the fitted polynomial function
            poly = np.poly1d(coefficients)
    
            # Step 4: Generate a smoothed curve
            # Create a range of x values to plot the smooth curve
            #longer_side = np.maximum(np.max(x) - np.min(x) + 1, np.max(y) - np.min(y) + 1)
            x_fit = np.linspace(np.min(x), np.max(x), 100) # number of points should be the same as longer side
            y_fit = poly(x_fit)
            print('horizontal fish')
            
        else:
        
            # Fit a second-order polynomial (quadratic) to the data
            coefficients = np.polyfit(y, x, 2)
    
            # Generate the fitted polynomial function
            poly = np.poly1d(coefficients)
    
            # Step 4: Generate a smoothed curve
            # Create a range of x values to plot the smooth curve
            #longer_side = np.maximum(np.max(x) - np.min(x) + 1, np.max(y) - np.min(y) + 1)
            y_fit = np.linspace(np.min(y), np.max(y), 100) # number of points should be the same as longer side
            x_fit = poly(y_fit)
            
            print('vertical fish')
            
        
        poly_points = []    
        # Step 5: Overlay the fitted curve on the original image            
        # Draw the fitted curve (overlay) on the image
        for i in range(len(x_fit)):
            x_coord = int(x_fit[i])
            y_coord = int(y_fit[i])
    
            # Draw a small circle to represent each point of the fitted curve
            if 0 <= x_coord < image.shape[1] and 0 <= y_coord < image.shape[0]:
                cv2.circle(image, (x_coord, y_coord), 1, (0, 255, 0), -1)  # Red color for the curve
                poly_points.append((y_coord, x_coord))
    
        # Step 6: Save the image with the fitted curve
        if save:
            cv2.imwrite('fit_2ndpoly_v1.png', image)
            
        poly_points = np.asarray(poly_points)
            
        return poly_points
            
