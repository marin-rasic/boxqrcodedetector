# Box QR Code Detector

Project for detecting Boxes with QR codes on them in real time using camera. Using YOLOv8 model for object detection.

# Usage

```
python main.py <weights_file> [options]
```

Weights file should be a PyTorch file made for YOLOv8 model.

Avaliable options are:
```
--width <int>
    Width of images
    default: 600
--height <int>
    Height of images
    default: 480
-c, --confidence <float>
    Confidence threshold
    default: 0.8
```

For best results, image dimensions should match dimension of images used for training the model.

