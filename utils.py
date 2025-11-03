import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter
import pyrealsense2 as rs
from ultralytics import YOLO
import cv2
import numpy as np
from skimage.morphology import skeletonize, binary_closing, binary_opening, convex_hull_image
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter
from color_results import ColorResults
from depth_results import DepthResults
from scipy.ndimage import convolve
from skimage.graph import route_through_array
import matplotlib.pyplot as plt

def process_mask(mask):
    '''process mask to more natual fish shape'''
    mask = mask > 0
    mask = binary_closing(mask, footprint=np.ones((5,5), dtype=bool))
    mask = binary_opening(mask, footprint=np.ones((3,3), dtype=bool))
    #mask = convex_hull_image(mask)
    
    return mask


def find_endpoints(skeleton):
    """
    Detects endpoints in the skeleton (pixels with exactly 1 neighbor).
    """
    kernel = np.array([[1, 1, 1],
                       [1,10, 1],
                       [1, 1, 1]])
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    endpoints = (neighbor_count == 11) & (skeleton == 1)
    return np.column_stack(np.where(endpoints))  # Returns (row, col) pairs


def extract_longest_path_dijkstra(skeleton):
    """
    Finds the longest shortest path between any pair of endpoints in the skeleton.
    Uses Dijkstra’s algorithm to follow curved skeleton paths.
    Returns a binary mask of the longest path.
    """
    endpoints = find_endpoints(skeleton)
    if len(endpoints) < 2:
        return np.zeros_like(skeleton, dtype=bool)  # Not enough points to make a path

    # Convert skeleton into a cost map (0 = impassable, 1 = passable)
    cost_array = np.where(skeleton, 1, np.inf)

    max_len = 0
    best_path_coords = None

    # Try all unique pairs of endpoints
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            start = tuple(endpoints[i])
            end = tuple(endpoints[j])

            try:
                # Find the shortest path along skeleton
                path, cost = route_through_array(cost_array, start, end, fully_connected=True)
                if len(path) > max_len:
                    max_len = len(path)
                    best_path_coords = path
            except:
                # If no valid path exists (disconnected), skip
                continue

    # Create a binary mask for the best (longest) path
    if best_path_coords:
        path_mask = np.zeros_like(skeleton, dtype=bool)
        for r, c in best_path_coords:
            path_mask[r, c] = True
        return path_mask
    else:
        return np.zeros_like(skeleton, dtype=bool)  # No valid path found


def process_mask_to_centerline(binary_mask):
    """
    Full pipeline: skeletonize the mask, find longest centerline (curved), return as mask.
    """
    skeleton = skeletonize(binary_mask)
    longest_path_mask = extract_longest_path_dijkstra(skeleton)
    return skeleton, longest_path_mask

def track_points(curve_pixels, color_image):
    '''find start and end points and track the point'''
    #skeleton = (255*skeleton).astype(np.uint8)
    # Find nonzero pixel coordinates (white pixels of the curve)
    #curve_pixels = np.column_stack(np.where(skeleton == 255))
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
    for x,y in curve_pixels:
        print(x,y)
        print(len(adj_list[(x,y)]))
    endpoints = [p for p in adj_list if len(adj_list[p]) == 1]

    if len(endpoints) != 2:
        print("Error: The curve is not a simple open curve!")
        #print('end points: ', endpoints)
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
        #skeleton = np.stack((skeleton,skeleton,skeleton), axis=-1)
        cv2.circle(color_image, (start[1], start[0]), 1, (0,255,0), -1)
        cv2.circle(color_image, (end[1], end[0]), 1, (0,255,0), -1)
        cv2.imwrite('end_point.jpg', color_image)
        
        return ordered_curve
        

#calculate distance
def calculate_distance(point1, point2, depth_frame, depth_intrin):
    depth_p1 =  depth_frame.get_distance(point1[0], point1[1])#depth_image[point1]       
    point1_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, point1, depth_p1)#*depth_scale)      
    depth_p2 =  depth_frame.get_distance(point2[0], point2[1])#depth_image[point2]       
    point2_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, point2, depth_p2)#*depth_scale)        
    size_ = np.sqrt(np.sum(np.square(np.asarray(point1_3d) - np.asarray(point2_3d))))
        
    return size_


def click_event_auto(event, x, y, flags, param):
    bg_removed_img, depth_frame, depth_intrin, masks, color_image, cam = param
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb_results = ColorResults(masks, color_image)
        img = rgb_results.color_image.copy()
        #for single_poly in rgb_results.polys:
            #ordered_curve = track_points(single_poly, img)
        print('2d process finished')
        d_results = DepthResults(depth_frame, rgb_results, cam, 'poly') #line type is poly, line or skeleton
        fishes_lengths = d_results.get_lengths()
        
        
        #todo filter depth map and add across x axis


# function to display the coordinates of 
# of the points clicked on the image  
def click_event_semi(event, x, y, flags, param): 
 
    # checking for left mouse clicks 
    img, depth_frame, depth_intrin = param
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        global point1
        point1 = (x,y)
  
        # displaying the coordinates 
        # on the image window 
        cv2.circle(img, (x,y), radius=1, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image', img) 
        
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        point2 = (x,y)
  
        # displaying the coordinates 
        # on the image window 
        cv2.circle(img, point2, radius=1, color=(0, 255, 255), thickness=-1)
        cv2.imshow('image', img) 
        distance = calculate_distance(point1, point2, depth_frame, depth_intrin) 
        text = str(distance*1000)[:3]+'mm'
        text_loc = (int((point1[0] + point2[0])/2), int((point1[1] + point2[1])/2))
        cv2.line(img, point1, point2, (255,255,255),1)
        cv2.putText(img, text, text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, color=(0,0,255), thickness=1)         
        cv2.imshow('image', img) 
        cv2.imwrite('fishsize.jpg', img)

def align_frame(align, frames):
    '''align depth and color frame, then return them'''
    aligned_frames = align.process(frames)
    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    aligned_color_frame = aligned_frames.get_color_frame()
    
    return depth_frame, aligned_color_frame
    
def remove_bg(depth_image, color_image, clipping_distance, grey_color=137):
    '''remove points too far and negative depth, return color image with valid depth values'''
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image) #
    
    return bg_removed
    


def skeletonize_with_smooth(mask, skeleton_settings={'iter_num':1, 'threshold':70,}):
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
    height, width = image.shape[0], image.shape[1] 
    y_min = max(0, min(height - 1, y_min))
    y_max = max(0, min(height - 1, y_max))
    
    # Draw the fitted line on the original image
    cv2.line(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red line
    cv2.imwrite('fit_line_v1.png', image)
    
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
    cv2.imwrite('fit_2ndpoly_v1.png', image)
    
    
def save_skeleton(skeleton, mask, image_with_mask, image_with_skeleton):
    '''save skeleton and mask'''
    mask = np.stack((mask,mask,mask), axis=-1)
    skeleton = np.stack([skeleton] * 3, axis=-1)
    image_with_mask = 255*mask + image_with_mask
    image_with_skeleton = 255*skeleton + image_with_skeleton
        
    cv2.imwrite('overlap_xl_v1.png', image_with_mask)
    cv2.imwrite('sk_xl_v1.png', image_with_skeleton) 
    print('save results.....')
    
    return image_with_mask, image_with_skeleton 
    
    
def find_inconsistent_points(depth_image, poly_points, valid_locations, consistence_threshold):
    '''find inconsistent depth locations, valid locations are locations after filtering too far or too close points'''
    init_valid_ind = np.nonzero(valid_locations)
    init_valid_coords = poly_points[init_valid_ind]
    init_valid_depth = depth_image[init_valid_coords[:,0], init_valid_coords[:,1]]
    average_depth = np.mean(init_valid_depth)
    for i in range(poly_points.shape[0]):
        if abs(depth_image[poly_points[i,0], poly_points[i,1]] - average_depth) > consistence_threshold:
            valid_locations[i] = 0
            
    return valid_locations
    
    




