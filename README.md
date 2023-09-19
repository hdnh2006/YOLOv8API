<div align="center">
  <img width="450" src="assets/Flask_logo.svg">
</div>

# Yolov8 Flask API for detection and segmentation

<a href="https://www.buymeacoffee.com/hdnh2006" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

![Screen GIF](assets/screen.gif)

This code is based on the YOLOv8 code from Ultralytics and it has all the functionalities that the original code has:
- Different source: images, videos, webcam, RTSP cameras.
- All the weights are supported: TensorRT, Onnx, DNN, openvino.

The API can be called in an interactive way, and also as a single API called from terminal and it supports all the tasks provided by YOLOv8 (detection, segmentation, classification and pose estimation) in the same API!!!!

All [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png">

## Requirements

Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed, including `torch>=1.7`. To install run:

```bash
$ pip3 install -r requirements.txt
```

## [Detection](https://docs.ultralytics.com/tasks/detect), [Segmentation](https://docs.ultralytics.com/tasks/segment), [Classification](https://docs.ultralytics.com/tasks/classify) and [Pose Estimation](https://docs.ultralytics.com/tasks/pose) models pretrained on the [COCO](https://docs.ultralytics.com/datasets/detect/coco) in the same API

`predict_api.py` can deal with several sources and can run into the cpu, but it is highly recommendable to run in gpu.

```bash
Usage - sources:
    $ python predict_api.py --weights yolov8s.pt --source 0                              # webcam
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
    $ python predict_api.py --weights yolov8s.pt                # PyTorch
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

Usage - tasks:
    $ python predict_api.py --weights yolov8s.pt                # Detection
                                     yolov8s-seg.pt             # Segmentation
                                     yolov8s-cls.pt             # Classification
                                     yolov8s-pose.pt            # Pose Estimation
```

## Interactive implementation implemntation

You can deploy the API able to label an interactive way.

Run:

```bash
$ python detect_api.py --device cpu # to run into cpu (by default is gpu)
```
Open the application in any browser 0.0.0.0:5000 and upload your image or video as is shown in video above.


## How to use the API

### Interactive way
Just open your favorite browser and go to 0.0.0.0:5000 and intuitevely load the image you want to label and press the buttom "Upload image".

The API will return the image or video labeled.

### Call from terminal or python program
The `client.py` code provides several example about how the API can be called. A very common way to do it is to call a public image from url and to get the coordinates of the bounding boxes:

```python
import requests

resp = requests.get("http://0.0.0.0:5000/predict?source=https://atlassafetysolutions.com/wp/wp-content/uploads/2019/06/ppe.jpeg&save_txt=T",
                    verify=False)
print(resp.content)

```
And you will get a json with the following data:

```
b'{"results": [{"name": "person", "class": 0, "confidence": 0.9284878969192505, "box": {"x1": 174.23370361328125, "y1": 6.4221954345703125, "x2": 1014.720458984375, "y2": 1053.7127685546875}}, {"name": "refrigerator", "class": 72, "confidence": 0.3910899758338928, "box": {"x1": 697.6565551757812, "y1": 0.0, "x2": 1372.587646484375, "y2": 1056.9697265625}}]}'
```


## About me and contact

This code is based on the YOLOv8 code from Ultralytics and it has been modified by Henry Navarro
 
If you want to know more about me, please visit my blog: [henrynavarro.org](https://henrynavarro.org).
