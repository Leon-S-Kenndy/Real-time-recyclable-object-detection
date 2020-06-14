# Real-time-recyclable-object-detection
# Table of Contents

- [Introduction](#introduciton)
- [Tutorial](#tutorial)
    - [Description](#Description)
    - [Project structure](#project-structure)
    - [Complie Darkent on Linux](#complie-darkent-on-linux)
    - [Label Image](#label-image)
    - [Begin train](#Begin-train)
- [Now result](#Now-result)

# Introduction

This project involves the development of a tool for detecting (in real-time) which household waste can be recycled. **The rule for recycling is based on Loughborough University's rule.**

The goals for this projects are:

1. A well trained deep neural network which is based on tiny-yolo to recognise recyclable items in real-time via a camera.
2. An simple app based on this well well trained deep neural network.

# Tutorial

## Description

This part is a tutorial to teach you how to use the YOLO object detector to detect objects. In this project, the obejects are household wastes. According to Loughborough University's recycling rule we have 9 classes, batteries, Cans_Tins, Cardboard, cups, Glass, Paper, Plastics, FoodWaste, GeneralWaste.

For more details, look at [rules](https://www.charnwood.gov.uk/pages/green_recycling_bin) and [Right Stuff, Right Bin .pdf ](https://www.lboro.ac.uk/media/wwwlboroacuk/content/facilitiesmanagement/downloads/intranetdept/fmhealthsafety/Right%20Stuff,%20Right%20Bin%20.pdf)

## Project structure

This is part of project structure:

    .
    ├── darknet.data
    ├── data
    │   ├── obj
    |   ├── obj.data
    │   ├── obj.names
    │   ├── test.txt
    │   └── train.txt
    ├── weights
    ├── yolov3-tiny.cfg
    └── yolov3-tiny.conv.15

* `datknet.data` : Configuration for training
* `data\obj` : Directory to save images
* `obj.data` : Configuration for training
* `data\obj.names` : List with object names
* `data\test.txt` : List of image filenames for testing
* `data\train.txt` : List of image filenames for training
* `weights` : Directory to save trained weights
* `yolov3-tiny.cfg` : Structure of yolov3-tiny
* `yolov3-tiny.conv.15` : Pre-trained weight



### Complie Darkent on Linux
I recommend AlexeyAB's Darknet version because he added correct calculation of mAP, F1, IoU, Precision-Recall and can draw chart of average-Loss and accuracy-mAP during training and so many other things.

I simply copy some guidlines from his github to teach you how to install darknet on linux. For more deatails, look at [Darkent](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-cmake)

To complie, just do make in the Real_time_recyclable_object_detection directory in command line 
Before make, you can set such options in the  `Makefile`: [link](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/Real_time_recyclable_object_detection/Makefile)
* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `DEBUG=1` to bould debug version of Yolo
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. 

After `make`, try this command in command line in the Real_time_recyclable_object_detection directory
```sh
$ ./darknet imtest data/eagle.jpg
```
If you get a bunch of windows with eagles in them you've succeeded! 

### Label image

I use VOTT as the tool to label images. VOTT is an open-source annotation and labeling tool for image and video assets.

To use VOTT, download it from https://github.com/microsoft/VoTT/releases and choose version 1.7.2, because only VOTT 1 can export data in YOLO format. 

I use VOTT on windows so here I only take windows VOTT as a example to show how to label images. 

Once you havb downloade VOTT, just do as the follow steps (these steps are based on VOTT1 github repository and change slight for this project. For more details, please check [VOTT](https://github.com/Microsoft/VoTT/tree/v1)):

1.Put collected images into `data\obj`. 

2.Create `compress.py` file and insert the following code:

```
import os
import glob
from PIL import Image

def thumbnail_pic(path):
    path_save = "output/"
    a = glob.glob(r'*.jpg')
    for x in a:
        name = os.path.join(path, x)
        im = Image.open(name)
        im.thumbnail((416, 416))
        print(im.format, im.size, im.mode)
        name = os.path.join(path_save, x)
        im.save(name, 'JPEG')
    print('Done!')

if __name__ == '__main__':
    path = '.'
    thumbnail_pic(path)
```

This wil help you to change the size of image to 416 X 416. This will help you save time during training. `416 X 416` is defined in  yolov3-tiny.cfg as input width and height. You can change this to the size you like.

3. Open VOTT, select the option to tag an image directory, which is the red rectangle part in the image. 
![STEP1](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-1.png)
4. Configure the tagging job and specify the settings, the red rectangle part in the image is the name of the class that you want to label
![STEP2](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-2.png)
    **Tagging Region Type**:  type of bounding box for tagging regions<br>
      - *Rectangle*: tag bounding boxes of any dimension
      - *Square*: tag bounding boxes of auto-fixed dimensions

    **Labels**: labels of the tagged regions (e.g. `Cat`, `Dog`, `Horse`, `Person`)<br>
5. Tag each Image
    ![STEP3](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-3.png)
    **Tagging**: click and drag a bounding box around the desired area, then move or resize the region until it fits the object
     - Selected regions appear as red ![red](https://placehold.it/15/f03c15/000000?text=+) and unselected regions will appear as blue ![#1589F0](https://placehold.it/15/1589F0/000000?text=+).
     - Assign a tag to a region by clicking on it and selecting the desired tag from the labeling toolbar at the bottom of the tagging control
     - Click the ![cleartags](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/cleartags.png) button to clear all tags on a given frame
    **Navigation**: you can navigate between video frames by using the ![prev-nxt](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/prev-next.png) buttons, the left/right arrow keys, or the video skip bar
     - Tags are auto-saved each time a frame is changed
6.Export Image directory Tags using the Object Detection Menu or Ctrl/Cmd + E
    ![VOTT5](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-5.png)
    *Note on exporting: the tool reserves a random 20% sample of the tagged frames as a test set.*
    Specify the following export configuration settings:
    ![VOTT5](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-5.png)
    - **Export Format**: What framework to export to defaults to *CNTK*<br>
    - **Export Frames Until**: how far into the video the export operation will proceed<br>
      - *Last Tagged Region*: exports frames up until the last frame containing tags
      - *Last Visited Frame*: exports frames up until the last frame that the user explicitly visited
      - *Last Frame*: exports all video frames<br>
    - **Output directory**: directory path for exporting training data<br>
    



## Begin train

1. Open file `yolov3-tiny.cfg` and:

  * change line batch to `batch=64`
  * change line subdivisions to `subdivisions=32`
  * change line max_batches to `classes*2000`, in this project it will be `18000` 
  * change line steps to 80% and 90% of max_batches, in this project it will be `steps=14400,16200`
  * change line `classes=9` to your number of objects in each of `[yolo]`-layers:
  * change [`filters=42`] to filters=(classes + 5)x3 in the `[convolutional]` before each `[yolo]` layer

2. Download default weights file for yolov3-tiny: https://pjreddie.com/media/files/yolov3-tiny.weights

3. Get pre-trained weights `yolov3-tiny.conv.15` using command: `darknet.exe partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15`

4. Start training by using the command line: 
     
   To train on Linux use command: 
    ```
    ./darknet detector train /home/leon/Downloads/Real_time_recyclable_object_detection/darknet2.data /home/leon/Downloads/Real_time_recyclable_object_detection/yolov3-tiny.cfg /home/leon/Downloads/Real_time_recyclable_object_detection/yolov3-tiny.conv.15 > /home/leon/Downloads/Real_time_recyclable_object_detection/trainInt.log -map
    ```
# Now result


can run the original yolov3-tiny on Android to detect photos
(Android settings NDK16 √ NDK21 ×)

To-do:

move the customed model to Android

provide settings for users when they face different rubbish bin

to detect video in real-time

try to improve model

