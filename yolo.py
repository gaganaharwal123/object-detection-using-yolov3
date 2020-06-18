import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
FLAGS = []
def live_feed(obj_dec):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    parser.add_argument('--download-model',
        type=bool,
        default=False,
        help='Set to True, if the model weights and configurations \
                are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False,
        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()


    labels = open('darknet\\data\\coco.names').read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet('darknet\\cfg\\yolov3.cfg', 'darknet\\yolov3.weights')

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    #net = cv.dnn.readNetFromDarknet('darknet\\cfg/yolov3_custom.cfg', 'darknet/yolov3.weights')
    count=0
    vid = cv.VideoCapture(0)
    while True:
        _, frame = vid.read()
        height, width = frame.shape[:2]

        if count == 0:
            frame, boxes, confidences, classids, idxs = infer_image(net,  obj_dec, layer_names, \
                                height, width, frame, colors, labels, FLAGS)
            count += 1
        else:
            frame, boxes, confidences, classids, idxs = infer_image(net,  obj_dec, layer_names, \
                                height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
            count = (count + 1) % 6

        cv.imshow('webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()
def imag_det(img_path,obj_dec):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    parser.add_argument('--download-model',
        type=bool,
        default=False,
        help='Set to True, if the model weights and configurations \
                are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False,
        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

    labels = open('darknet\\data\\obj.names').read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet('darknet/cfg/yolov3_custom.cfg', 'darknet/backup/yolov3_custom_last.weights')

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    labels = open('darknet\\data\\obj.names').read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet('darknet/cfg/yolov3_custom.cfg', 'darknet/backup/yolov3_custom_last.weights')

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    try:
        img = cv.imread(img_path)
        height, width = img.shape[:2]
    except:
        raise 'Image cannot be loaded!\n\
                            Please check the path provided!'
    finally:
        img, _, _, _, _ = infer_image(net, obj_dec, layer_names, height, width, img, colors, labels, FLAGS)
        
        cv.imwrite('image_out.jpg',img)
def vid_det(vid_path,obj_dec):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')
                
    parser.add_argument('-vo', '--video-output-path',
        type=str,
                default='./output.avi',
        help='The path of the output video file')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    parser.add_argument('--download-model',
        type=bool,
        default=False,
        help='Set to True, if the model weights and configurations \
                are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False,
        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

    labels = open('darknet\\data\\obj.names').read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet('darknet/cfg/yolov3_custom.cfg', 'darknet/backup/yolov3_custom_last.weights')

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    labels = open('darknet\\data\\obj.names').read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet('darknet/cfg/yolov3_custom.cfg', 'darknet/backup/yolov3_custom_last.weights')

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    try:
        vid = cv.VideoCapture(vid_path)
        height, width = None, None
        writer = None
    except:
        raise 'Video cannot be loaded!\n\
                            Please check the path provided!'

    finally:
        while True:
            grabbed, frame = vid.read()
            # Checking if the complete video is read
            if not grabbed:
                break

            if width is None or height is None:
                height, width = frame.shape[:2]

            frame, _, _, _, _ = infer_image(net, obj_dec, layer_names, height, width, frame, colors, labels, FLAGS)

            if writer is None:
                # Initialize the video writer
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
                                (frame.shape[1], frame.shape[0]), True)


            writer.write(frame)

        print ("[INFO] Cleaning up...")
        writer.release()
        vid.release()
        cap = cv.VideoCapture('output.mp4')
        main()
def play_videoFile(filePath,mirror=False):
 
    cap = cv.VideoCapture(filePath) 
    frame_counter = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_counter += 1
        #If the last frame is reached, reset the capture and the frame_counter
        if frame_counter == cap.get(cv.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        # Our operations on the frame come here
       
        # Display the resulting frame
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
 
def main():
    play_videoFile('output.avi',mirror=False)
 