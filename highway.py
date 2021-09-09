"""# **Run a Pre-Trained Detectron2 Model**
Prepare the Mask-RCNN PreTrained Model (based on COCO Dataset)
"""
import time
import os
import detectron2
import math
import torch
import cv2
import numpy as np
# from google.colab.patches import cv2.imshow
import matplotlib.pyplot as plt
from os.path import basename, splitext
from time import sleep
from scipy.ndimage import rotate as rot
from detectron2.structures.boxes import Boxes
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

setup_logger()
start_t = time.time()

# Create a Detectron2 Config
cfg = get_cfg()

# Add project specific configuration (e.g. TensorMask) - if you are not running a model in core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.NMS_THRESH_TEST = 0.90
cfg.MODEL.DEVICE = "cpu"

# *****************************
# The Trained Predictor at COCO
# *****************************
predictor = DefaultPredictor(cfg)

end_t = time.time()
elapsed_time = end_t - start_t
print('Finished with Mask-RCNN Model Preparation -- Ready for Predictions !' + "\n *** Elapsed Time = " + str(
    elapsed_time) + " (msecs) ***")

"""Apply the **Mask-RCNN** to Find Cars in Parking Space Frames
Clean up frames folder: Remove old MAKSED_ files
"""

"""# ***The IMAGE-DIFF Code***"""
MAIN_FOLDER = './inputs'
# VIDEOS_FOLDER = MAIN_FOLDER + '/Videos'
OUTPUT_FRAMES_PATH = MAIN_FOLDER + '/frames'
one_frame_each = 24

# os.system('rm - f {OUTPUT_FRAMES_PATH} / masked *')
# os.system('rm -f {OUTPUT_FRAMES_PATH}/diff_*.png')

# img_0 = cv2.imread(OUTPUT_FRAMES_PATH + '/frame0.png')
# img_1 = cv2.imread(OUTPUT_FRAMES_PATH + '/frame' + str(one_frame_each) + '.png')
# # Frame Difference
# frame_diff = cv2.absdiff(img_1, img_0)
# cv2.imwrite(OUTPUT_FRAMES_PATH + '/diff_0_' + str(one_frame_each) + '.png', frame_diff)
#
# previous = img_1
# counter = 2 * one_frame_each
#
# while True:
#     frame_fname = OUTPUT_FRAMES_PATH + '/frame' + str(counter) + '.png'
#
#     img = cv2.imread(frame_fname)
#
#     if img is None:
#         # Couldn't Read Image
#         print('Could Not Read Frame ... %s !' % frame_fname)
#         break
#
#     # Frame Difference
#     frame_diff = cv2.absdiff(img, previous)
#
#     bname = '/diff_' + str(counter - one_frame_each) + '_' + str(counter) + '.png'
#     full_bname = OUTPUT_FRAMES_PATH + bname
#     # print(full_bname)
#
#     cv2.imwrite(full_bname, frame_diff)
#
#     previous = img
#     counter += one_frame_each

# os.system('cp / content / parking - space / Videos / frames / diff_ *.png / content / drive / MyDrive / ML - apps / '
#           'parking - space - monitoring / Videos / frames /')

"""Define a function that is responsible for model application ... **Per Frame**
# **Section MASK-RCNN Application to Low-Freq Frames**
"""

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

"""Utility Function [1]: ***Associate a Car BBox with a Parking-Slot***"""


def detect_parking_slot(bbox_car, centroid=None):
    bbox_car_x_median = None
    bbox_car_y_median = None
    if centroid is not None:
        bbox_car_x_median = centroid[0]
        bbox_car_y_median = centroid[1]
    else:
        # TOP LEFT COORDS
        bbox_car_tl_x = int(bbox_car[0])
        bbox_car_tl_y = int(bbox_car[1])

        # LOW RIGHT COORDS
        bbox_car_lr_x = int(bbox_car[2])
        bbox_car_lr_y = int(bbox_car[3])

        bbox_car_x_median = (bbox_car_tl_x + bbox_car_lr_x) / 2.0
        bbox_car_y_median = (bbox_car_tl_y + bbox_car_lr_y) / 2.0

    # print("Bounding Box Median Point = (%f , %f)" % (bbox_car_x_median, bbox_car_y_median))

    dist_bbox_slot = 10000
    row_num = 0
    slot_num = 0

    for lane in lanes:  # WILL HAVE TO CHANGE TO LANES
        driving_lane = int(lane[0])
        parking_slot_at_row = int(lane[1])

        slot_corner_1 = lane[2]
        if 'None' not in slot_corner_1:
            sc_1 = slot_corner_1.split('-')
            # print("sc_1 = " + " , ".join(sc_1))

            sc01_x = int(sc_1[0].strip())
            sc01_y = int(sc_1[1].strip())
        else:
            sc01_x = 0
            sc01_y = 0

        slot_corner_2 = lane[3]
        if 'None' not in slot_corner_2:
            sc_2 = slot_corner_2.split('-')
            # print("sc_2 = " + " , ".join(sc_2))

            sc02_x = int(sc_2[0].strip())
            sc02_y = int(sc_2[1].strip())
        else:
            sc02_x = 0
            sc02_y = 0

        slot_corner_3 = lane[4]
        if 'None' not in slot_corner_3:
            sc_3 = slot_corner_3.split('-')
            # print("sc_3 = " + " , ".join(sc_3))

            sc03_x = int(sc_3[0].strip())
            sc03_y = int(sc_3[1].strip())
        else:
            sc03_x = 0
            sc03_y = 0

        slot_corner_4 = lane[5]
        if 'None' not in slot_corner_4:
            sc_4 = slot_corner_4.split('-')
            # print("sc_4 = " + " , ".join(sc_4))

            sc04_x = int(sc_4[0].strip())
            sc04_y = int(sc_4[1].strip())
        else:
            sc04_x = 0
            sc04_y = 0

        slot_x_median = (sc01_x + sc02_x + sc03_x + sc04_x) / 4.0
        slot_y_median = (sc01_y + sc02_y + sc03_y + sc04_y) / 4.0
        # print( "Median Point of Slot %d is (%f, %f)" % (parking_slot_at_row, slot_x_median, slot_y_median) )

        dist_x_2 = math.pow(bbox_car_x_median - slot_x_median, 2)
        dist_y_2 = math.pow(bbox_car_y_median - slot_y_median, 2)
        dist_bbox_slot_c = math.sqrt(dist_x_2 + dist_y_2)

        if dist_bbox_slot_c < dist_bbox_slot:
            dist_bbox_slot = dist_bbox_slot_c

            row_num = driving_lane
            slot_num = parking_slot_at_row

    return row_num, slot_num, dist_bbox_slot


# Function with inputs two arrays with {Labels, BBoxes}
# Send for Processing Each .... Instance {One-Label, One-BBox}
#
def locate_instances(labels_arr, bboxes_arr):
    lines_arr = []
    slots_arr = []
    dists_arr = []

    vehicles_num = 0
    for i in range(len(labels_arr)):
        label = labels_arr[i]

        if label in ['car', 'motorcycle', 'bus', 'bike', 'truck']:
            bbox_obj = bboxes_arr[i]
            vehicles_num += 1

            # print( bbox_car.__class__ )  --> ndarray
            # print( bbox_car ) --> [317.80377   59.471947 438.29037  113.94816 ]

            row_num, slot_num, dist = detect_parking_slot(bbox_obj)

            print("Found Car in Parking Space --> Row = %d :: Slot = %d" % (row_num, slot_num))

            lines_arr.append(row_num)
            slots_arr.append(slot_num)
            dists_arr.append(dist)

    return lines_arr, vehicles_num, dists_arr


"""# ***Parking-Frame*** Apply Object Detector / Frame
Utility Function [2]: Needs to be called from Frames-Folder Reader
"""


def process_parking_frame(frame_fname):
    # Read the file and import
    # Parking-Space image at time XXX ...
    img = cv2.imread(frame_fname)

    # cv2.imshow(img)

    # ***********************************
    # Apply Mask-RCNN Model ... Find Cars
    # ***********************************
    outputs = predictor(img)

    # Information #01 --> Bounding Boxes of Interesting objects found
    # ---------------------------------------------------------------
    outputs_pred_boxes = outputs["instances"].pred_boxes
    # num_boxes = len(outputs_pred_boxes)
    # print("Number of INTERESTING objects found in '%s' = %d" % (frame_fname, num_boxes))

    bboxes_arr = []
    for i in outputs_pred_boxes.__iter__():
        bbox_i = i.cpu().numpy()
        # For Each Object an Array --> [x_left_up, y_left_up, x_right_down, y_right_down]
        bboxes_arr.append(bbox_i)

    # Information #02 --> Labels of interesting objects
    outputs_pred_classes = outputs["instances"].pred_classes

    classes_arr = []
    for i in outputs_pred_classes.__iter__():
        class_index = i.cpu().numpy()
        my_class_name = class_names[class_index]
        if class_index not in [2, 3, 5, 6, 7]:
            print('Non-Vehicle Object detected in the Highway')  # FIRST EXCEPTION POINT
            break
        # print('--> %d ... %s' % (i, my_class_name))
        classes_arr.append(my_class_name)

    # Print labels of all contained objects
    # classes_str = " , ".join(classes_arr)
    # print(classes_str) --> car , car , car , car

    ### !!! LOCATE THE PARKING SLOT

    frame_history = []
    diff_history = []
    lane_history = []
    for lane in lanes:
        img_crop = img[lane]
        lane_objects = predictor(img_crop)
        lane_classes = lane_objects["instances"].pred_classes
        vehicles = [j for j in range(len(lane_classes)) if lane_classes[j] in [2, 3, 5, 6, 7]]
        lane_cars_number = len(lane_objects["instances"].pred_boxes[vehicles])  # keep #vehicles in lane
        lane_history.append(lane_cars_number)
    frame_history.append(lane_history)

    for k in range(len(frame_history) - 1):
        l1 = frame_history[k]
        l2 = frame_history[k + 1]
        for l1_j, l2_j in zip(l1, l2):
            diff_history.append(l1_j - l2_j)


    lines, vehicles_num, dists = locate_instances(classes_arr, bboxes_arr)
    ### !!! Found Parking Slot !!! ##

    # Draw predictions in input image
    # -------------------------------
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    # for instance in outputs["instances]:
    # print(instance)

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    if 'frame1104.png' in frame_fname:
        cv2.imshow(out.get_image()[:, :, ::-1])

        # tmp = out.get_image()[:, :, ::-1]
        # plt.imsave("%s/%s" % (OUTPUT_FRAMES_PATH, '/masked_frame0.png'), tmp, cmap=plt.cm.gray)

    my_basename = basename(frame_fname)
    my_basename = splitext(my_basename)[0]
    my_masked_filename = "%s/masked_%s.png" % (OUTPUT_FRAMES_PATH, my_basename)
    # print(my_masked_filename)

    # cv2.imwrite("%s/masked_%s.png" % (OUTPUT_FRAMES_PATH, frame_fname), out.get_image()[:, :, ::-1])

    tmp = out.get_image()[:, :, ::-1]
    plt.imsave(my_masked_filename, tmp, cmap=plt.cm.gray)

    frame_num = int(my_basename.split('frame')[1])
    return frame_num, bboxes_arr, outputs_pred_classes.cpu().numpy(), lines, vehicles_num, dists