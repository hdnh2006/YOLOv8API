<div align="center">
  <img width="450" src="assets/Flask_logo.svg">
</div>

# Yolov8 Flask API for detection and segmentation

## Requirements

Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed, including `torch>=1.7`. To install run:

```bash
$ pip3 install -r requirements.txt
```

## Object detection API

`detect_api.py` can deal with several sources and can run into the cpu, but it is highly recommendable to run in gpu.

```bash
Usage - sources:
    $ python detect_api.py --weights yolov5s.pt --source 0                               # webcam
                                                         img.jpg                         # image
                                                         vid.mp4                         # video
                                                         screen                          # screenshot
                                                         path/                           # directory
                                                         list.txt                        # list of images
                                                         list.streams                    # list of streams
                                                         'path/*.jpg'                    # glob
                                                         'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                         'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python detect_api.py --weights yolov5s.pt                 # PyTorch
                                     yolov5s.torchscript        # TorchScript
                                     yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                     yolov5s_openvino_model     # OpenVINO
                                     yolov5s.engine             # TensorRT
                                     yolov5s.mlmodel            # CoreML (macOS-only)
                                     yolov5s_saved_model        # TensorFlow SavedModel
                                     yolov5s.pb                 # TensorFlow GraphDef
                                     yolov5s.tflite             # TensorFlow Lite
                                     yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                     yolov5s_paddle_model       # PaddlePaddle
```


## Instance segmentation API

`segment_api.py` can deal with several sources and can run into the cpu, but it is highly recommendable to run in gpu.

```bash
Usage - sources:
    $ python segment_api.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python segment_api.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
```

## Interactive implementation implemntation

You can deploy an API able to label an interactive way. So what you have to do is

Run:

```bash
$ python detect_api.py --device cpu # to run into cpu (by default is gpu)
```
Open the application in any browser 0.0.0.0:5000.

If you prefer, you can launch the docker container application, running:

```bash
$ docker build -t yoloAPI --no-cache .
$ docker run --name yoloAPI yoloAPI:latest
```
Check the container network:

```bash
$ docker inspect yoloAPI
```
And open the application with the container ip, for example 172.17.0.2:5000.

## How to use the API

Just open your favorite browser and go to 0.0.0.0:5000 and intuitevely load the image you want to label and press the buttom "Upload image".

The API will return the image labeled.


## About me and contact

This code is based on the YOLOv5 from ultralytics and it has been modified by Henry Navarro
 
If you want to know more about me, please visit my blog: henrynavarro.org.
