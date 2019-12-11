# Real-time-recyclable-object-detection
# Table of Contents

- [Introduction](#introduciton)
- [Tutorial](#tutorial)
    - [Description](##Description)
    - [Requirements](###Requirements)
        - [Darknet](#####Darknet)
        - [Yolo_mark](#####Yolo_mark)
- [Usage](#usage)
    - [Generator](#generator)
- [Badge](#badge)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

# Introduction

This project involves the development of a tool for detecting (in real-time) which household waste can be recycled. **The rule for recycling is based on Loughborough University's rule.**

The goals for this projects are:

1. A well trained deep neural network which is based on tiny-yolo to recognise recyclable items in real-time via a camera.
2. An simple app based on this well well trained deep neural network.

# Tutorial

## Description

This part is a tutorial to teach you how to use the YOLO object detector to detect objects. In this project, the obejects are household wastes. According to Loughborough University's recycling rule we have 9 classes, batteries, Cans_Tins, Cardboard, cups, Glass, Paper, Plastics, FoodWaste, GeneralWaste.

For more details, look at [rules](https://www.charnwood.gov.uk/pages/green_recycling_bin) and [Right Stuff, Right Bin .pdf ](https://www.lboro.ac.uk/media/wwwlboroacuk/content/facilitiesmanagement/downloads/intranetdept/fmhealthsafety/Right%20Stuff,%20Right%20Bin%20.pdf)

## Requirements

We need to prepare several tools before we begin.
### [* Darkent(AlexeyAB)](https://github.com/AlexeyAB/darknet)
I recommend AlexeyAB's version because he added correct calculation of mAP, F1, IoU, Precision-Recall and can draw chart of average-Loss and accuracy-mAP during training and so many other things.
I simply copy some guidlines from his github to teach you how to install darknet on linux. For more deatails, look at [Darkent](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-cmake)
```sh
$ git clone https://github.com/AlexeyAB/darknet.git
$ cd darknet
$ make
```
Before make, you can set such options in the `Makefile`
* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `DEBUG=1` to bould debug version of Yolo
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
    or use in such a way: `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov3.cfg yolov3.weights test.mp4`
* `ZED_CAMERA=1` to build a library with ZED-3D-camera support (should be ZED SDK installed), then run
    `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov3.cfg yolov3.weights zed_camera`
After `make`, try this command
```sh
$ ./darknet imtest data/eagle.jpg
```
If you get a bunch of windows with eagles in them you've succeeded! 

### * [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)

## Usage

This is only a documentation package. You can print out [spec.md](spec.md) to your console:

```sh
$ standard-readme-spec
# Prints out the standard-readme spec
```

### Generator

To use the generator, look at [generator-standard-readme](https://github.com/RichardLitt/generator-standard-readme). There is a global executable to run the generator in that package, aliased as `standard-readme`.

## Badge

If your README is compliant with Standard-Readme and you're on GitHub, it would be great if you could add the badge. This allows people to link back to this Spec, and helps adoption of the README. The badge is **not required**.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

To add in Markdown format, use this code:

```
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
```

## Example Readmes

To see how the specification has been applied, see the [example-readmes](example-readmes/).

## Related Efforts

- [Art of Readme](https://github.com/noffle/art-of-readme) - 💌 Learn the art of writing quality READMEs.
- [open-source-template](https://github.com/davidbgk/open-source-template/) - A README template to encourage open-source contributions.

## Maintainers

[@RichardLitt](https://github.com/RichardLitt).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/RichardLitt/standard-readme/issues/new) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

### Contributors

This project exists thanks to all the people who contribute. 
<a href="graphs/contributors"><img src="https://opencollective.com/standard-readme/contributors.svg?width=890&button=false" /></a>


## License

[MIT](LICENSE) © Richard Littauer
