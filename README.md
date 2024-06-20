# ONNX_Runtime_Web_Yolov8-seg_Batching
This repository demonstrates how to use ONNX Runtime to run Yolov8-seg models in the browser, including support for batched image processing. The example application displays several images and applies the Yolov8-seg model to detect objects and segment them, with results displayed directly on the webpage.

## Features

- **Object Detection:** Detects various objects in images using YOLOv8.
- **Segmentation:** Provides segmentation masks for specific objects like persons.
- **Real-time Processing:** Processes images in real-time directly in the web browser.
- **Interactive Gallery:** Displays processed images with interactive hover effects.

## Technologies Used

- **ONNX Runtime:** For running the YOLOv8 and segmentation models.
- **TensorFlow.js:** For image preprocessing and manipulation.
- **OpenCV.js:** Used for drawing masks.
- **HTML/CSS/JavaScript:** Frontend development technologies.

## Before Segmentation

![Before Segmentation](https://github.com/shimaamorsy/ONNX_Runtime_Web_Yolov8-seg_Batching/blob/main/img/before.PNG)

This image shows the state before segmentation.

  ## After Segmentation

![After Segmentation](https://github.com/shimaamorsy/ONNX_Runtime_Web_Yolov8-seg_Batching/blob/main/img/after.png)

This image shows the segmentation results of the web application.

- ## Installation
 Simply clone the repository and open `index.html` in live server in a web browser .

 ```bash
git clone https://github.com/yourusername/ONNX_Runtime_Web_Yolov8-seg_Batching.git


