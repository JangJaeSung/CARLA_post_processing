#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import copy
import os.path
import time

import carla

from carla import ColorConverter as cc

import cv2
import re

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (0, 0, 255)
WBB_COLOR = (255, 0, 0)
BB_White = (255, 255, 255)

Vehicle_COLOR = np.array([142, 0, 0])
Walker_COLOR = np.array([60, 20, 220])

rgb_info = np.zeros((540, 960, 3), dtype="i")
seg_info = np.zeros((540, 960, 3), dtype="i")

def reading_data(index):
    global rgb_info, seg_info

    data = []
    w_data = []
    k = 0
    w = 0

    rgb_img1 = cv2.imread('Image_rgb/rgb'+ str(index)+ '.png', cv2.IMREAD_COLOR)
    seg_img1 = cv2.imread('Image_seg/seg'+ str(index)+ '.png', cv2.IMREAD_COLOR)
    if str(rgb_img1) != "None" and str(seg_img1) != "None":
        # Vehicle
        with open('BB_info/test'+ str(index), 'r') as fin:
            bounding_box_rawdata = fin.read()

        bounding_box_data = re.findall(r"-?\d+", bounding_box_rawdata)
        line_length = len(bounding_box_data) / 16 

        bb_data = [[0 for col in range(8)] for row in range(line_length)] # need row range

        for i in range(len(bounding_box_data)/2):
            j = i*2
            data.append(tuple((int(bounding_box_data[j]), int(bounding_box_data[j+1]))))

        for i in range(len(bounding_box_data)/16):
            for j in range(8):
                bb_data[i][j] = data[k]
                k += 1

        # Walker
        with open('WBB_info/test'+ str(index), 'r') as w_fin:
            w_bounding_box_rawdata = w_fin.read()

        w_bounding_box_data = re.findall(r"-?\d+", w_bounding_box_rawdata)
        w_line_length = len(w_bounding_box_data) / 16 

        w_bb_data = [[0 for col in range(8)] for row in range(w_line_length)] # need row range

        for i in range(len(w_bounding_box_data)/2):
            j = i*2
            w_data.append(tuple((int(w_bounding_box_data[j]), int(w_bounding_box_data[j+1]))))

        for i in range(len(w_bounding_box_data)/16):
            for j in range(8):
                w_bb_data[i][j] = w_data[w]
                w += 1


        rgb_info = rgb_img1
        seg_info = seg_img1
        return bb_data, line_length, w_bb_data, w_line_length 

    else:
        return False

# Vehicle
def eight_to_four(bounding_boxes, line_length):
    points_array = []
    bb_4data = [[0 for col in range(4)] for row in range(line_length)]
    k = 0
    for i in range(line_length):
        points_array_x = []
        points_array_y = []      
        for j in range(8):
            points_array_x.append(bounding_boxes[i][j][0])
            points_array_y.append(bounding_boxes[i][j][1])

            max_x = max(points_array_x)
            min_x = min(points_array_x)
            max_y = max(points_array_y)
            min_y = min(points_array_y)           

        points_array.append(tuple((min_x, min_y)))
        points_array.append(tuple((max_x, min_y)))
        points_array.append(tuple((max_x, max_y)))
        points_array.append(tuple((min_x, max_y)))

    for i in range(line_length):
        for j in range(len(points_array)/line_length):
            bb_4data[i][j] = points_array[k]
            k += 1  

    return bb_4data

# Walker
def w_eight_to_four(bounding_boxes, w_line_length):
    points_array = []
    w_bb_4data = [[0 for col in range(4)] for row in range(w_line_length)]
    k = 0
    for i in range(w_line_length):
        points_array_x = []
        points_array_y = []      
        for j in range(8):
            points_array_x.append(bounding_boxes[i][j][0])
            points_array_y.append(bounding_boxes[i][j][1])

            max_x = max(points_array_x)
            min_x = min(points_array_x)
            max_y = max(points_array_y)
            min_y = min(points_array_y)           

        points_array.append(tuple((min_x, min_y)))
        points_array.append(tuple((max_x, min_y)))
        points_array.append(tuple((max_x, max_y)))
        points_array.append(tuple((min_x, max_y)))

    for i in range(w_line_length):
        for j in range(len(points_array)/w_line_length):
            w_bb_4data[i][j] = points_array[k]
            k += 1  

    return w_bb_4data




def check_out_of_range(points):
    for x in range(4):
        if points[x][0] > 0 and points[x][0] < VIEW_WIDTH and points[x][1] > 0 and points[x][1] < VIEW_HEIGHT:
            continue
        else:
            return False

    return True   

def get_points_x_tight(x1, x2, range_min, range_max, color):
    global seg_info
    state = False
    cali_point = 0
    if (x1 < x2):
        for search_point in range(x1, x2):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][0] == color:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(x1, x2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][0] == color:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point



def get_points_y_tight(y1, y2, range_min, range_max, color):
    global seg_info
    state = False
    cali_point = 0
    if (y1 < y2):
        for search_point in range(y1, y2):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][0] == color:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(y1, y2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][0] == color:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

def small_objects_excluded(min_x, max_x, min_y, max_y, bb_min):
    diff_x = max_x - min_x
    diff_y = max_y - min_y
    if (diff_x > bb_min and diff_y > bb_min):
        return True
    return False

def check_object_in_BB(min_x, max_x, min_y, max_y, color):
    global seg_info
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if seg_info[y, x][0] == color[0]:
                return True
    
    return False

def processing(img, data, w_data, index):
    global seg_info
    vehicle_class = 0
    walker_class = 1

    #f = open("custom_data/image"+str(index) + ".txt", 'w')
    for bbox in data:
        if check_out_of_range(bbox):
            min_x = bbox[0][0]
            max_x = bbox[1][0]
            min_y = bbox[0][1]
            max_y = bbox[2][1]
            #print(min_x, max_x, min_y, max_y)

            #if seg_info[center_y, center_x][0] == 142:
            if check_object_in_BB(min_x, max_x, min_y, max_y, Vehicle_COLOR) and  small_objects_excluded(min_x, max_x, min_y, max_y, 10):
                cali_min_x = get_points_x_tight(min_x, max_x, min_y, max_y, 142)
                cali_max_x = get_points_x_tight(max_x, min_x, min_y, max_y, 142)
                cali_min_y = get_points_y_tight(min_y, max_y, min_x, max_x, 142)
                cali_max_y = get_points_y_tight(max_y, min_y, min_x, max_x, 142)
                
                #print(cali_min_x, cali_max_x, cali_min_y, cali_max_y)

                darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
                darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
                darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
                darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

                #f.write(str(vehicle_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                #str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), BB_COLOR, 3)
                cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), BB_COLOR, 3)
                cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), BB_COLOR, 3)
                cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), BB_COLOR, 3)

    for wbbox in w_data:
        if check_out_of_range(wbbox):
            min_x = wbbox[0][0]
            max_x = wbbox[1][0]
            min_y = wbbox[0][1]
            max_y = wbbox[2][1]

            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            #if seg_info[center_y, center_x][0] == 60:
            if check_object_in_BB(min_x, max_x, min_y, max_y, Walker_COLOR) and small_objects_excluded(min_x, max_x, min_y, max_y, 7):
                cali_min_x = get_points_x_tight(min_x, max_x, min_y, max_y, 60)
                cali_max_x = get_points_x_tight(max_x, min_x, min_y, max_y, 60)
                cali_min_y = get_points_y_tight(min_y, max_y, min_x, max_x, 60)
                cali_max_y = get_points_y_tight(max_y, min_y, min_x, max_x, 60)

                darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
                darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
                darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
                darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

                #f.write(str(walker_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                #str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), WBB_COLOR, 3)
                cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), WBB_COLOR, 3)
                cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), WBB_COLOR, 3)
                cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), WBB_COLOR, 3)

    #f.close()




    #cv2.imwrite('test_automatic/a'+str(index)+'.png', img)
    cv2.imwrite('test_automatic/test'+str(index)+'.png', img)
    #cv2.imwrite('custom_data/image'+str(index)+'.png', img)

index_count = 0
def run():
    global rgb_info
    global index_count
    train = open("my_data/train.txt", 'w')
    for i in range(5000):
        if reading_data(i) != False:
            four_vertices_points = eight_to_four(reading_data(i)[0], reading_data(i)[1])
            w_four_vertices_points = w_eight_to_four(reading_data(i)[2], reading_data(i)[3])
            processing(rgb_info, four_vertices_points, w_four_vertices_points, i)
            train.write(str('custom_data/image'+str(i) + '.jpg') + "\n")
            index_count = index_count + 1
            print(i)
    train.close()
start = time.time()
run()
end = time.time()
print(index_count)
print(float(end - start))

#four_vertices_points = eight_to_four(reading_data(34)[0], reading_data(34)[1])

#processing(rgb_info, four_vertices_points)
#cv2.imshow('rgb_img', rgb_info)
#cv2.waitKey(0)
#cv2.destroyAllWindows()










####################################################
# Convert Image and Ground Truth Data to json File #
####################################################