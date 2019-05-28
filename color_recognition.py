# setup
import numpy as np
import argparse
import random
import time
import cv2 as cv
import os
from sklearn.cluster import KMeans


from pythonosc import udp_client
from pythonosc import osc_message_builder
from pythonosc import osc_bundle_builder

import utils

# class ColorDetector:
#     def __init__(self, labels, colors, net):
#         self.LABELS = 

def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    #     help="path to input image")
    ap.add_argument("-m", "--mask-rcnn",
        help="base path to mask-rcnn directory", default="mask-rcnn-coco")
    ap.add_argument("-v", "--visualize", type=int, default=0,
        help="whether or not we are going to visualize each instance")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="minimum threshold for pixel-wise mask segmentation")
    ap.add_argument("--clusters", type=int, default=3,
        help="# of clusters")
    ap.add_argument("--ip", default="127.0.0.1",
        help="The IP address of the OSC server")
    ap.add_argument("-p", "--port", type=int, default=5005,
        help="The port number of the OSC server")
    args = vars(ap.parse_args())
    return args

def find_instances(image):
    # Construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    # show timing information and volume information on Mask R-CNN
    print(f'[INFO] Mask R-CNN took {(end - start):.6f} seconds')
    print(f'[INFO] boxes shape: {boxes.shape}')
    print(f'[INFO] masks shape: {masks.shape}')

    return boxes, masks

def process_instance(box, mask):
    # clone our original image so we can draw on it
    clone = image.copy()

    # resize the box and mask
    box = box * np.array([W, H, W, H])
    (startX, startY, endX, endY) = box.astype("int")
    boxW = endX - startX
    boxH = endY - startY
    mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_NEAREST)
    mask = (mask > args["threshold"])

    # extract the ROI of the image
    roi = clone[startY:endY, startX:endX]

    # convert the mask from a boolean to an integer mask
    binaryMask = (mask * 255).astype("uint8")
    # apply the mask
    instance = cv.bitwise_and(roi, roi, mask=binaryMask)
    # get color cluster histogram and centers
    clusters = cluster_colors(instance)
    # sort clusters
    clusters = sorted(clusters, key=lambda x:x[0], reverse=True)
    clusters = list(filter(lambda x: x[0] > 0.3, clusters))
    # (hist, clt_centrs) = clusters

    # Build osc bundle and send to server
    msg = osc_message_builder.OscMessageBuilder(address='/colors')
    print(f"[INFO] Cluster Info:")
    for i in range(len(clusters)):
        msg.add_arg(clusters[i][1].astype(int).tolist())
        print(f'\t{(clusters[i][1].astype(int))}:({(clusters[i][0]):.2%})')
    client.send(msg.build())

    if args["visualize"] > 0:
        visualize(image, box, roi, instance, mask, clusters)

def cluster_colors(instance):
    # reshape the instance
    instance_flatten = instance.reshape(instance.shape[0] * instance.shape[1], 3)
    # remove black pixels
    instance_flatten = instance_flatten[instance_flatten[:, 0] != 0]

    # cluster the pixel instensities
    clt =KMeans(n_clusters=args["clusters"])
    clt_start = time.time()
    clt.fit(instance_flatten)
    clt_end = time.time()
    print(f"[INFO] Clustering takes {(clt_end - clt_start):.4f} seconds")
            
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
            
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    clt_centers = clt.cluster_centers_

    return list(zip(hist, clt_centers))

def get_color_bar(cluster_centers):
    # plot color bar
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for (percent, color) in cluster_centers:
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(endX), 0), (int(startX), 50), color.astype("uint8").tolist(), -1)
        startX = endX
    return bar


def visualize(img, box, roi, instance, mask, clusters):
    bar = get_color_bar(clusters)

    # cv.imshow("ROI", roi)
    # cv.imshow("Mask", binaryMask)
    # cv.imshow("Segmented", instance)
    # cv.imshow('Bar', bar)

    clone = image.copy()
    # extract *only* the masked region of the ROI by passing
    # in the boolean mask array as our slice condition
    roi =roi[mask]

    # randomly select a color
    color = random.choice(COLORS)
    blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

    # store the blended ROI in the original image
    (startX, startY, endX, endY) = box.astype("int")
    clone[startY:endY, startX:endX][mask] = blended

    # draw the bounding box of the instance on the image
    color = [int(c) for c in color]
    cv.rectangle(clone, (startX, startY), (endX, endY), color, 2)

    # draw predicted label and associated probability of the
    # instance segmentation on the image
    text = f'{LABELS[classID]}: {confidence:.4f}'
    cv.putText(clone, text, (startX, startY - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    clone[startY:startY+50,startX:startX+300] = bar

    cv.imshow("Output", clone)




########## Main Program ##########
args = parse_args()

# Setup osc client
client = udp_client.SimpleUDPClient(args["ip"], args["port"])

# load labels and colors
labelsPath = os.path.sep.join([args["mask_rcnn"], "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# load the set of colors that will be used when visualizing a given instance segmentation
colorsPath = os.path.sep.join([args["mask_rcnn"], 'colors.txt'])
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")

# derive the paths to the mask r-cnn weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
    "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
    "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load mask r-cnn trained on the coco dataset from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv.dnn.readNetFromTensorflow(weightsPath, configPath)

vid_cap = cv.VideoCapture(0)

video_process_start = time.time()
video_process_end = time.time()
while True:
    # print(f"elapse:{video_process_end - video_process_start}")
    video_process_end = time.time()

    key = cv.waitKey(1)
    if key == 27:
        break
    if video_process_end - video_process_start > 5.0:
        video_process_start = time.time()
    else:
        continue 
    
    # load input image and grab its spatial dimensions
    ret, image = vid_cap.read()
    (H, W) = image.shape[:2]

    boxes, masks = find_instances(image)

    # Loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # associated with the prediction
        classID = int(boxes[0,0,i,1])
        confidence = boxes[0,0,i,2]

        # filter out weak and non-person predictions
        if confidence > args["confidence"] and LABELS[classID] == 'person':
            box = boxes[0,0,i,3:7]
            mask = masks[i, classID]
            
            process_instance(box, mask)

    

cv.destroyAllWindows()
vid_cap.release()







        




