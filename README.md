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
    $ python detect_api.py --weights yolov8s.pt --source 0                               # webcam
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
    $ python detect_api.py --weights yolov8s.pt                 # PyTorch
                                     yolov8s.torchscript        # TorchScript
                                     yolov8s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                     yolov8s_openvino_model     # OpenVINO
                                     yolov8s.engine             # TensorRT
                                     yolov8s.mlmodel            # CoreML (macOS-only)
                                     yolov8s_saved_model        # TensorFlow SavedModel
                                     yolov8s.pb                 # TensorFlow GraphDef
                                     yolov8s.tflite             # TensorFlow Lite
                                     yolov8s_edgetpu.tflite     # TensorFlow Edge TPU
                                     yolov8s_paddle_model       # PaddlePaddle
```


## Instance segmentation API

`segment_api.py` can deal with several sources and can run into the cpu, but it is highly recommendable to run in gpu.

```bash
Usage - sources:
    $ python segment_api.py --weights yolov8s-seg.pt --source 0                               # webcam
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
    $ python segment_api.py --weights yolov8s-seg.pt                 # PyTorch
                                          yolov8s-seg.torchscript        # TorchScript
                                          yolov8s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov8s-seg_openvino_model     # OpenVINO
                                          yolov8s-seg.engine             # TensorRT
                                          yolov8s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov8s-seg_saved_model        # TensorFlow SavedModel
                                          yolov8s-seg.pb                 # TensorFlow GraphDef
                                          yolov8s-seg.tflite             # TensorFlow Lite
                                          yolov8s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov8s-seg_paddle_model       # PaddlePaddle
```

## Interactive implementation implemntation

You can deploy the API able to label an interactive way.

Run:

```bash
$ python detect_api.py --device cpu # to run into cpu (by default is gpu)
```
Open the application in any browser 0.0.0.0:5000 and upload your image or video as is shown in video:

![Screen GIF](assets/screen.gif)

## How to use the API

Just open your favorite browser and go to 0.0.0.0:5000 and intuitevely load the image you want to label and press the buttom "Upload image".

The API will return the image labeled.


## About me and contact

This code is based on the YOLOv8 from ultralytics and it has been modified by Henry Navarro
 
If you want to know more about me, please visit my blog: henrynavarro.org.
