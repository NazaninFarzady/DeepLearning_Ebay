# DeepLearning_Ebay
The main goal of project: Follow in the steps of Ebay and build an app that removes the background from product image
The sub-goal is to train model for changing the background of image

## Dataset
We consider using a subset of Coco dataset which is a popular large dataset in Image processing

## Models
We choosed some models to train in order to meet our goals including:
1. Tiramisu
2. Unet
3. DenseNet
4. DeepLab v3
5. Mask RCNN

After training and testing the models with dataset, we will do benchmarking and compair the models performance

## Web Interface
At the end we will provide a web application interface in order to able user for entering image, choosing from the models and select remove/change the background of the image

## Requirements
Python 3  
TensorFlow = 1.x  
GPU   

## Instalation 
1.Clone this repository.  
'''  
git clone https://github.com/NazaninFarzady/DeepLearning_Ebay.git  
'''  
2.Install dependencies.  
'''  
pip install -r requirements.txt   
'''  

## How-to
bg_removal.ipynb shows how to remove the background of your image.  
Change the path of your image.
Saved the image in png format.
