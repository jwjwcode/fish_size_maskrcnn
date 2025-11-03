import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter

def track_points(skeleton, i):
    '''find start and end points and track the point'''
    skeleton = (255*skeleton).astype(np.uint8)
    # Find nonzero pixel coordinates (white pixels of the curve)
    curve_pixels = np.column_stack(np.where(skeleton == 255))

    # Build a connectivity graph (adjacency list)
    from collections import defaultdict

    adj_list = defaultdict(list)
    pixel_set = set(map(tuple, curve_pixels))

    for x, y in curve_pixels:
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
                 (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]  # 8-connectivity

        for nx, ny in neighbors:
            if (nx, ny) in pixel_set:
                adj_list[(x, y)].append((nx, ny))

    # Find endpoints (nodes with only one neighbor)
    endpoints = [p for p in adj_list if len(adj_list[p]) == 1]

    if len(endpoints) != 2:
        print("Error: The curve is not a simple open curve!")
    else:
        start, end = endpoints

        # Trace the curve from start to end
        ordered_curve = [start]
        visited = set([start])

        while ordered_curve[-1] != end:
            current = ordered_curve[-1]
            for neighbor in adj_list[current]:
                if neighbor not in visited:
                    ordered_curve.append(neighbor)
                    visited.add(neighbor)
                    break

        # Print or save the ordered coordinates
        print("Ordered curve coordinates:", ordered_curve)
        print('start and end: ', start, end)
        skeleton = np.stack((skeleton,skeleton,skeleton), axis=-1)
        cv2.circle(skeleton, (start[1], start[0]), 1, (0,255,0), -1)
        cv2.circle(skeleton, (end[1], end[0]), 1, (0,255,0), -1)
        cv2.imwrite(str(i) + '_end_point.jpg', skeleton)
        
def detect_save_poly(image, skeleton):
    '''fit polynomial to the skeleton'''
    skeleton = (255*skeleton).astype(np.uint8)
    # Find the coordinates of the white pixels (skeleton points)
    skeleton_points = np.column_stack(np.where(skeleton == 255)) 

    # Step 3: Fit a second-order polynomial (quadratic) to the extracted points
    x = skeleton_points[:, 1]  # x-coordinates (columns of the image)
    y = skeleton_points[:, 0]  # y-coordinates (rows of the image)
    
    # Fit a second-order polynomial (quadratic) to the data
    coefficients = np.polyfit(x, y, 2)
    
    # Generate the fitted polynomial function
    poly = np.poly1d(coefficients)
    
    # Step 4: Generate a smoothed curve
    # Create a range of x values to plot the smooth curve
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = poly(x_fit)
    
    # Step 5: Overlay the fitted curve on the original image

    
    # Draw the fitted curve (overlay) on the image
    for i in range(len(x_fit)):
        x_coord = int(x_fit[i])
        y_coord = int(y_fit[i])
    
        # Draw a small circle to represent each point of the fitted curve
        if 0 <= x_coord < image.shape[1] and 0 <= y_coord < image.shape[0]:
            cv2.circle(image, (x_coord, y_coord), 1, (0, 255, 0), -1)  # Red color for the curve
    
    # Step 6: Save the image with the fitted curve
    cv2.imwrite('image_with_fitted_curve.png', image)
          


def skeletonize_with_smooth(mask, skeleton_settings):
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

    return skeleton
    
def detect_save_lines(image, skeleton):
    '''detect lines and show the end points'''
    # Apply linear regression to fit a line

    skeleton = (255*skeleton).astype(np.uint8)
    # Find the coordinates of the white pixels (skeleton points)
    skeleton_points = np.column_stack(np.where(skeleton == 255)) 
       
    # Perform least squares line fitting
    # Use np.polyfit to fit a line to the points
    # polyfit returns the coefficients [m, b] for y = mx + b
    m, b = np.polyfit(skeleton_points[:, 1], skeleton_points[:, 0], 1)
    # Calculate the x range of the skeleton points
    x_min = skeleton_points[:, 1].min()  # Minimum x-coordinate
    x_max = skeleton_points[:, 1].max()  # Maximum x-coordinate

    # Calculate the y-coordinates for the min and max x values
    y_min = int(m * x_min + b)  # y at x_min
    y_max = int(m * x_max + b)  # y at x_max

    # Ensure the y-coordinates are within image bounds
    height, width = image.shape[:2]
    y_min = max(0, min(height - 1, y_min))
    y_max = max(0, min(height - 1, y_max))
    
    # Draw the fitted line on the original image
    cv2.line(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red line
    cv2.imwrite('fit_line_v1.png', image)
  
def save_skeleton(skeleton, skeleton_settings, mask, image_with_mask, image_with_skeleton):
    '''save skeleton and mask'''
    mask = np.stack((mask,mask,mask), axis=-1)
    skeleton = np.stack([skeleton] * 3, axis=-1)
    image_with_mask = 255*mask + image_with_mask
    image_with_skeleton = 255*skeleton + image_with_skeleton
        
    cv2.imwrite(str(skeleton_settings['threshold']) + '_' + str(skeleton_settings['iter_num']) + '_overlap_xl_v1.png', image_with_mask)
    cv2.imwrite(str(skeleton_settings['threshold']) + '_' + str(skeleton_settings['iter_num']) +'_sk_xl_v1.png', image_with_skeleton) 
    
    return image_with_mask, image_with_skeleton 
    
def get_centerline(outputs, test_image, skeleton_settings = {'iter_num':1,'threshold':70,} ):
       
    # View results
    result = outputs["instances"]
    if result is None:
        print('no fish detected')
    else:   
        masks = result.pred_masks.cpu().numpy() # mask in matrix format (num_objects x H x W)
        print(masks.shape)
           
        #initialize masks and skeleton images to be saved
        image_with_mask = test_image
        image_with_skeleton = test_image
    
        #save mask for each object
        for i in range(masks.shape[0]):
            #mask = cv2.resize(masks[i,:,:], (640,480))  
            mask = masks[i,:,:] 
            skeleton = skeletonize_with_smooth(mask, skeleton_settings)
            track_points(skeleton, i)
            image_with_mask, image_with_skeleton = save_skeleton(skeleton, skeleton_settings, mask, image_with_mask, image_with_skeleton)
            #detect_save_lines(test_image, skeleton)
            detect_save_poly(test_image, skeleton)