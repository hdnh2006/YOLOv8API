<div align="center">
  <img width="450" src="assets/Flask_logo.svg">
</div>

# Yolov8 Flask API for detection and segmentation

<a href="https://www.buymeacoffee.com/hdnh2006" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

![Screen GIF](assets/screen.gif)

This code is based on the YOLOv5 from Ultralytics and it has all the functionalities that the original code has:
- Different source: images, videos, webcam, RTSP cameras.
- All the weights are supported: TensorRT, Onnx, DNN, openvino.

The API can be called in an interactive way, and also as a single API called from terminal. 



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
Open the application in any browser 0.0.0.0:5000 and upload your image or video as is shown in video above.


## How to use the API

### Interactive way
Just open your favorite browser and go to 0.0.0.0:5000 and intuitevely load the image you want to label and press the buttom "Upload image".

The API will return the image or video labeled.

### Call from terminal or python program
The `client.py` code provides several example about how the API can be called. A very common way to do it is to call a public image from url and to get the coordinates of the bounding boxes:

```python
import requests

resp = requests.get("http://0.0.0.0:5000/detect?url=https://atlassafetysolutions.com/wp/wp-content/uploads/2019/06/ppe.jpeg&save_txt=T",
                    verify=False)
print(resp.content)

```
And you will get a json with the following data:

```
b'{"results": [{"class": 72, "x": 0.647187, "y": 0.495779, "w": 0.421875, "h": 0.991557, "conf": null}, {"class": 0, "x": 0.371563, "y": 0.497655, "w": 0.525625, "h": 0.982176, "conf": null}]}'
```


## About me and contact

This code is based on the YOLOv5 from Ultralytics and it has been modified by Henry Navarro
 
If you want to know more about me, please visit my blog: henrynavarro.org.
