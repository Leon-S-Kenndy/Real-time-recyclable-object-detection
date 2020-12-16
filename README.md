# Real-time-recyclable-object-detection
# Table of Contents

- [Introduction](#introduciton)
- [Tutorial](#tutorial)
    - [Description](#Description)
    - [Model training part](#model-training-part)
     - [Compile Darkent on Linux](#complie-darkent-on-linux)
     - [Label Image](#label-image)
     - [Begin training](#begin-training)
    - [App Part](#app-part)
     - [1. Project Construction](#1-project-construction) 
         - [1.1 Source Import](#11-source-import) 
     - [2. Source code modification](#2-source-code-modification)
         - [2.1 include folder](#21-include-folder)
         - [2.2 src folder](#22-src-folder)
     - [3. Cmake file configuration](#3-cmake-file-configuration)
     - [4. asserts folder configuration](#4-asserts-folder-configuration)
     - [5. JNI Interface Configuration](#5-jni-interface-configuration)
     - [6. Java Configuration](#6-java-configuration)
- [Now result](#now-result)
 - [Download APP](#download-app)
 - [The performance of model](#the-performance-of-model)
 - [Custom model](#custom-model)

# Introduction

This project involves the development of a tool for detecting (in real-time) which household waste can be recycled. **The rule for recycling is based on Loughborough University's rule.**

The goals for this project are:

1. A well trained deep neural network which is based on tiny-yolo to recognize recyclable items in real-time via a camera.
2. A simple app based on this well well trained deep neural network.

# Tutorial

## Description

This part is a tutorial to teach you how to use the YOLO object detector to detect objects and how to move the model to the app. In this project, the objects are household wastes. According to Loughborough University's recycling rule, we have 9 classes, batteries, Cans_Tins, Cardboard, cups, Glass, Paper, Plastics, FoodWaste, GeneralWaste.

For more details, look at [rules](https://www.charnwood.gov.uk/pages/green_recycling_bin) and [Right Stuff, Right Bin .pdf ](https://www.lboro.ac.uk/media/wwwlboroacuk/content/facilitiesmanagement/downloads/intranetdept/fmhealthsafety/Right%20Stuff,%20Right%20Bin%20.pdf)

## Model training part
This part is based on AlexeyAB's Github repository.
### Compile Darkent on Linux
I recommend AlexeyAB's Darknet version because he added correct calculation of mAP, F1, IoU, Precision-Recall and can draw a chart of average-Loss and accuracy-mAP during training and so many other things.

I simply copy some guidelines from his Github to teach you how to install darknet on Linux. For more details, look at [Darkent](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-cmake)

To compile, just do make in the Real_time_recyclable_object_detection directory in the command line 
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

To use VOTT, download it from https://github.com/microsoft/VoTT/releases and choose version 1.7.2 because only VOTT 1 can export data in YOLO format. 

I use VOTT on windows so here I only take windows VOTT as an example to show how to label images. 

Once you have download VOTT, just do as the following steps (these steps are based on VOTT1 GitHub repository and change slight for this project. For more details, please check [VOTT](https://github.com/Microsoft/VoTT/tree/v1)):

1. Open VOTT, select the option to tag an image directory, which is the red rectangle part in the image. 
![STEP1](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-1.png)

2. Configure the tagging job and specify the settings, the red rectangle part in the image is the name of the class that you want to label
![STEP2](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-2.png)
    **Tagging Region Type**:  type of bounding box for tagging regions<br>
      - *Rectangle*: tag bounding boxes of any dimension
      - *Square*: tag bounding boxes of auto-fixed dimensions

    **Labels**: labels of the tagged regions (e.g. `Cat`, `Dog`, `Horse`, `Person`)<br>

3. Tag each Image
    ![STEP3](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-3.png)
    **Tagging**: click and drag a bounding box around the desired area, then move or resize the region until it fits the object
     - Selected regions appear as red ![red](https://placehold.it/15/f03c15/000000?text=+) and unselected regions will appear as blue ![#1589F0](https://placehold.it/15/1589F0/000000?text=+).
     - Assign a tag to a region by clicking on it and selecting the desired tag from the labeling toolbar at the bottom of the tagging control
     - Click the ![cleartags](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/cleartags.png) button to clear all tags on a given frame
    **Navigation**: you can navigate between video frames by using the ![prev-nxt](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/prev-next.png) buttons, the left/right arrow keys, or the video skip bar
     - Tags are auto-saved each time a frame is changed

4.Export Image directory Tags using the Object Detection Menu or Ctrl/Cmd + E
    ![VOTT5](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-4.png)
    
    *Note on exporting: the tool reserves a random 20% sample of the tagged frames as a test set.*
    Specify the following export configuration settings:
    
    ![VOTT5](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/VOTT-5.png)
    - **Export Format**: What framework to export to defaults to *CNTK*<br>
    - **Export Frames Until**: how far into the video the export operation will proceed<br>
      - *Last Tagged Region*: exports frames up until the last frame containing tags
      - *Last Visited Frame*: exports frames up until the last frame that the user explicitly visited
      - *Last Frame*: exports all video frames<br>
    - **Output directory**: directory path for exporting training data<br>
 
5.The export iamge directory looks like:

    .
    ├── data
    │   ├── obj
    |   ├── obj.data
    │   ├── obj.names
    │   ├── test.txt
    │   └── train.txt
    
* `data\obj` : Directory to save images
* `data\obj.data` : Configuration for training
* `data\obj.names` : List with object names
* `data\test.txt` : List of image filenames for testing
* `data\train.txt` : List of image filenames for training


6.Create `compress.py` file in `data\` and insert the following code:

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


### Begin training

1. Here I use yolov3-tiny, so just open file `Real_time_recyclable_object_detection/cfg/yolov3-tiny.cfg` and:

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
   ./darknet detector train [directory]/obj.data directory/yolov3.cfg directory/yolov3-tiny.conv.15 -dont_show -mjpeg_port 8090 -map
    ```
    
    change [directory] to your own directory
    
## APP Part
This part is based on huuuuusy's github repository.
### 1. Project Construction

#### 1.1 Source Import
Build a project that supports C in Andorid Studio and download the source code from [Darknet] (https://github.com/pjreddie/darknet).

Create a new darknet folder in the project project's cpp folder, and copy the example, include, src folders in the downloaded source code to the android project.

And change the NDK version to NDK16.
### 2. Source code modification

#### 2.1 include folder

The header file of the darknet is placed under the include folder

#### 2.2 src folder

The src folder is placed in the darknet source. First, delete the compare.c file (compare.c has no header file, which does not work for the compilation of the entire library. If you do not delete it, compare.c will have a pointer problem at compile time. ). Then modify the image.c file, change the label path in the 232-line load_alphabet() function to sdcard/yolo/data/labels (this is the absolute path that will be placed on the phone later, if you don't make changes, then there will be a problem that the labels cannot be imported, resulting in a flashback problem).

### 3. Cmake file configuration

For the Cmake configuration information in this project, please refer to the CMakeLists.txt file in the code.

### 4. Asserts folder configuration

Place the cfg,data file in the darknet source code in the assert folder of the project.

### 5. JNI Interface Configuration

This project mainly modifies the darknetlib.c file. The image test code is taken from the official code example of darknet/examples/dector.c Line562~Line626. Please refer to the project code and comments for details.

### 6. Java Configuration

Modify the Yolo.java under the java folder to complete the relevant configuration.

# Now result

## Screenshot of runing APP
![Danalyse](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/Danalyse (1).jpg)
## Download APP

I provide an APK file in `https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/YOLOV3-on-Android/YoloRecycle.apk`, you can use it to download the app.

## The performance of model
Now, our best model is Yolov3-tiny-c4, the mAP of it is 83.51%. The average precision for each class is shown in the following picture:
![result](https://github.com/Leon-S-Kenndy/Real-time-recyclable-object-detection/blob/master/doc/images/result.png)

## Custom model

In the `custom` directory, I put the cfg file and the wights that I have tired, you can use it and the performance of each model can be seen in my [report](https://www.overleaf.com/read/kmjjbhmkqcgy).


