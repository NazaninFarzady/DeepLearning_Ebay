# DeepLearning_Ebay
The main goal of project: Follow in the steps of Ebay and build an app that removes the background from product image
The sub-goal is to train model for changing the background of image

## Dataset
We create our dataset similar to the coco structure with limited categories for our project

## Models
We choose Mask RCNN to train our model. The Mask R-CNN model generates bounding boxes and segmentation masks for each instance of an object in the image. Details on the requirements and background removal results for this repository can be found at the end of the document.

## Web Interface
At the end we will provide a web application interface in order to able user for entering image, choosing from the models and select remove/change the background of the image

## Requirements
- Python 3  
- TensorFlow > 1.3.0
- opencv-python
- skimage, matplotlib, numpy

## Instalation 
1. Clone this repository.  
```
git clone https://github.com/NazaninFarzady/DeepLearning_Ebay.git  
``` 
2. Install dependencies.  
```
pip install -r requirements.txt   
```

## RUN
[bg_removal.ipynb](https://github.com/NazaninFarzady/DeepLearning_Ebay/blob/TrainingModel/bg_removal.ipynb) shows how to remove the background of your image.  
Before run bg_removal.ipynb, download the [pre-trained model weights](https://wetransfer.com/downloads/2d56c023b813d61d1145b44a94b8ffe620200427132536/0cbf9373e6b9c610fa1e63385598143820200427132536/42cbf7) and change the path of your root and image.  

## Results
![image](https://github.com/NazaninFarzady/DeepLearning_Ebay/blob/TrainingModel/ebay_dataset_1/test/images/00000006.jpg)
![image](https://github.com/NazaninFarzady/DeepLearning_Ebay/blob/TrainingModel/detect_results/image_name.png)
