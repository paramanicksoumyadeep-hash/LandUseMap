# Delhi LandUse Analysis 
## Acknowledgements
I acknowledge that I have taken the help of the AI tool **ChatGPT** to implement some tasks, such as assigning image labels using the dominant (mode) land-cover class, filtering satellite images, and implementing the ResNet18 model. However, I have a **conceptual understanding** of how the code works and the purpose of the libraries, methods, and functions used.  

---

## Overview
This project focuses on classifying satellite images using deep learning techniques,trained on **ResNet18** model. 
Kagggle Dataset Link:-https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed

---

## Features
- 1)Plots the Delhi-NCR shapefile using matplotlib and filters sattelite images that fall under delhi region
    uses geopandas spatia; filters to filter the sattelite imagery area that is under delhi ncr map.
- 2)createimagecoordinates.py creates cororditanes for each image from the rbp image data directory.This is useful to map
    the images with their latitutde and longitude
- 3)For each image co-ordinate, created a land-cover patch from land_cover.tif using it's co-ordinate.Using Rasterio to fetch the .tif file
- 4)every patch is assigned a label with dominant (mode) land-cover class and assigned a land use category.
- 5)Implement a **ResNet18** convolutional neural network for image classification.with 60/40 train-test split percentage.
    The model was trained for 10 epochs.
- 6)Evaluate model performance using **accuracy** and **F1 score** metrics.  
  7)Plot Confusion Matrix to find where it fails mostly
---

## Results
- **Model Used:** ResNet18  
- **Accuracy:** 92.6%  
- **F1 Score:** 91.6%  
- Trained on a Kaggle-provided satellite imagery dataset.  

---

## Libraries Used
## Libraries Used
- geopandas  
- matplotlib  
- shapely  
- numpy  
- pandas  
- os  
- rasterio  
- torch  
- scikit-learn  
- torchvision  
- scipy  
- seaborn  
- pillow  

---



## How to Run
1. Clone this repository.  
2. Install the required libraries listed above using `pip install`.
3. Prepare the dataset (satellite imagery) in the designated folder.Name the folder data and add
   rgb folder, delhi_ncr_region.geojson, delhi_airshed.geojson, worldcover_bbox_delhi_ncr_2021.tif.in it
4.Run the script createimagecordinates.py
5. Run the main script to preprocess images, assign labels, and train the ResNet18 model.  
6. Evaluate the model using accuracy and F1 score metrics.  

---

## Future Work
- Extend the model to handle multi-class land-cover classification.  
- Optimize preprocessing and augmentation for larger satellite datasets.  
- Explore deployment of trained models for real-time environmental monitoring.  

---

