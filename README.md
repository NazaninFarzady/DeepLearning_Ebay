# DeepLearning_Ebay
The main goal of project: Follow in the steps of Ebay and build an app that removes the background from product image
The sub-goal is to train model for changing the background of image

## Dataset
We consider using a subset of Coco dataset which is a popular large dataset in Image processing

## Models
We choose Mask RCNN to train our model. The Mask R-CNN model generates bounding boxes and segmentation masks for each instance of an object in the image. Details on the requirements and background removal results for this repository can be found at the end of the document.

## Web Interface
At the end we will provide a web application interface in order to able user for entering image, choosing from the models and select remove/change the background of the image

## Requirements
- Python 3  
- TensorFlow > 1.3.0
- opencv-python
- skimage, matplotlib

## Instalation 
1.Clone this repository.  
```
git clone https://github.com/NazaninFarzady/DeepLearning_Ebay.git  
``` 
2.Install dependencies.  
```
pip install -r requirements.txt   
```

## RUN
[bg_removal.ipynb](https://github.com/NazaninFarzady/DeepLearning_Ebay/blob/TrainingModel/bg_removal.ipynb) shows how to remove the background of your image.  
Before run bg_removal.ipynb, download the pre-trained model weights /logs/object/mask_rcnn_objrct_0010.h5 on www.wetransfer.com and change the path of your root and image. 

## Results
