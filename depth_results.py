import numpy as np
import cv2


class DepthResults():
    '''class to process depth and 3D information for measuring size'''
    def __init__(self, depth_frame, rgb_results, cam, curve_type):
        '''initialize with rgb results and color image'''
        self.rgb_results = rgb_results
        if curve_type == 'poly':
            self.curves = self.rgb_results.polys
        elif curve_type == 'line':
            self.curves = self.rgb_results.lines
        elif curve_type == 'skeleton':
            self.curves = self.rgb_results.skeletons_points            
            #print('before:', self.curves[0])        
            #self.curves = self.sort_coords()        
            #print('after:', self.curves[0])
        
        self.depth_frame = depth_frame
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.cam = cam
        
    def sort_coords(self):
        '''sort the coords based on x first then y'''
        self.sorted_curves_coords = []
        for curve in self.curves:
            sorted_indices = np.lexsort((curve[:,0], curve[:,1]))
            sorted_curve_coords = curve[sorted_indices]
            self.sorted_curves_coords.append(sorted_curve_coords)
            
        return self.sorted_curves_coords
        
    def remove_invalid_depth_on_lines(self, max_depth_in_meters=5, min_depth_in_meters=0.0, consistence_threshold_in_meters=0.5):
        '''remove negative depth, depth too far, and inconsistent depth points on the fish line,
        return location mask where 1 represent valid and 0 represent invalid'''
        from utils import find_inconsistent_points
        max_distance = max_depth_in_meters / self.cam.depth_scale
        min_distance = min_depth_in_meters / self.cam.depth_scale
        consistence_threshold = consistence_threshold_in_meters / self.cam.depth_scale
        self.valid_poly_locations = [] #polys after removing invalid points
        for single_poly_points in self.curves:#loop through all polys
            single_poly_valid_locations = np.ones(single_poly_points.shape[0])
            for i in range(single_poly_points.shape[0]):#loop through all points on a poly poly coordinates is (row_ind, col_ind) 
                #print('..........', self.depth_image, self.depth_image.shape)
                #print(',,,,,,',single_poly_points, single_poly_points.shape)
                #print(';;;;;', self.depth_image[100][200])
                if self.depth_image[single_poly_points[i,0], single_poly_points[i,1]] <= min_distance or self.depth_image[single_poly_points[i,0], single_poly_points[i,1]] >= max_distance:
                    single_poly_valid_locations[i] = 0
            single_poly_valid_locations = find_inconsistent_points(self.depth_image, single_poly_points, single_poly_valid_locations, consistence_threshold)
            print('valid points percentage:', np.sum(single_poly_valid_locations) / single_poly_valid_locations.shape[0])                 
            self.valid_poly_locations.append(single_poly_valid_locations)
            
        return self.valid_poly_locations
        
    def get_valid_polys_coords(self, save=True):
        '''save valid poly points on rgb image and depth'''
        color_image = self.rgb_results.color_image.copy()
        depth_image = self.depth_image.copy()
        polys = self.curves
        valid_poly_locations = self.remove_invalid_depth_on_lines()
        self.valid_poly_coords = []
        for i, single_poly_valid_locations in enumerate(valid_poly_locations):#loop through all polys
            single_valid_coord_masked = single_poly_valid_locations[:,np.newaxis] * polys[i] #the invalid coords are set to 0.
            single_valid_coord = [] #list with valid coords only
            for i in range(single_valid_coord_masked.shape[0]):
                if np.sum(single_valid_coord_masked[i]) != 0:#extract x and y when both coordinates are not 0.
                    x_coord, y_coord = int(single_valid_coord_masked[i][1]), int(single_valid_coord_masked[i][0])
                    single_valid_coord.append((x_coord, y_coord))
                    #print(x_coord, y_coord)
                    cv2.circle(color_image, (x_coord, y_coord), 1, (0, 255, 0), -1)
                    depth_image[depth_image>1000] = 1000
                    depth_image = cv2.normalize(depth_image, None, 0,255,cv2.NORM_MINMAX)
                    depth_image = np.uint8(depth_image)
                    depth_image = cv2.applyColorMap(depth_image,cv2.COLORMAP_JET)
                    cv2.circle(depth_image, (x_coord, y_coord), 1, (255, 255, 255), -1)
                    
            self.valid_poly_coords.append(single_valid_coord)
        if save:                    
            cv2.imwrite('valied_lines_rgb.png', color_image)   
            cv2.imwrite('valied_lines_depth.png', depth_image) 
            
        return self.valid_poly_coords 
        
    def get_lengths(self, save=True):
        '''sum all the points to get the total length'''
        color_image = self.rgb_results.color_image.copy()
        valid_polys_coords = self.get_valid_polys_coords()
        self.fishes_lengths = []
        from utils import calculate_distance
        for single_valid_coord in valid_polys_coords:
            single_fish_length = 0
            for i in range(1,len(single_valid_coord)):
                point1 = single_valid_coord[i-1]
                point2 = single_valid_coord[i]
                segment = calculate_distance(point1, point2, self.depth_frame, self.depth_intrin)
                single_fish_length += segment
                cv2.circle(color_image, (point1[0], point1[1]), 1, (0, 255, 0), -1)
            # visualize the length
            text =  str(single_fish_length*1000)[:3]+'mm'
            mid_point_x = int((single_valid_coord[0][0] + single_valid_coord[-1][0]) / 2)
            mid_point_y = int((single_valid_coord[0][1] + single_valid_coord[-1][1]) / 2)
            position = (mid_point_x, mid_point_y)
            cv2.putText(color_image, text, position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,0,0),thickness=1,lineType=cv2.LINE_AA) 
               
            self.fishes_lengths.append(single_fish_length)
        if save:
            cv2.imwrite('length_results.png', color_image)
        print('fish length: ', self.fishes_lengths)
        print('3d process finished.')
        return self.fishes_lengths

            
            
        
        
            
     #todo test the remove invalid points, count percetage of removed points, interpolate removed points
     #try length with ske and line
                    

                
                
            
            
        
