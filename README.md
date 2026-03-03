# Delhi LandUse Analysis 

## Acknowledgements
I acknowledge that I have taken the help of the AI tool **ChatGPT** to implement some tasks, such as assigning image labels using the dominant (mode) land-cover class, filtering satellite images, and implementing the ResNet18 model. However, I have a **conceptual understanding** of how the code works and of the purpose of the libraries, methods, and functions used.  

---

## Overview
This project focuses on classifying satellite images using deep learning techniques, trained on the **ResNet18** model.  

**Kaggle Dataset Link:** [Earth Observation - Delhi Airshed](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed)

---

## Features
1. Plots the Delhi-NCR shapefile using Matplotlib and filters satellite images that fall under the Delhi region.  
   Uses GeoPandas spatial filters to select the satellite imagery area that is within the Delhi-NCR map.  
2. `createimagecoordinates.py` creates coordinates for each image from the RGB image data directory.  
   This is useful to map the images with their latitude and longitude.  
3. For each image coordinate, a land-cover patch is created from `land_cover.tif` using its coordinates, using Rasterio to fetch the `.tif` file.  
4. Each patch is assigned a label with the dominant (mode) land-cover class and a land-use category.  
5. Implements a **ResNet18** convolutional neural network for image classification with a 60/40 train-test split.  
   The model was trained for 10 epochs.  
6. Evaluates model performance using **accuracy** and **F1 score** metrics.  
7. Plots a confusion matrix to identify where the model performs poorly.  

---

## Results
- **Model Used:** ResNet18  
- **Accuracy:** 92.6%  
- **F1 Score:** 91.6%  
- Trained on a Kaggle-provided satellite imagery dataset.  

---

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
3. Prepare the dataset (satellite imagery) in the designated folder. Name the folder `data` and include:  
   - `rgb/` folder  
   - `delhi_ncr_region.geojson`  
   - `delhi_airshed.geojson`  
   - `worldcover_bbox_delhi_ncr_2021.tif`  
4. Run the script `createimagecoordinates.py`.  
5. Run the main script to preprocess images, assign labels, and train the ResNet18 model.  
6. Evaluate the model using accuracy and F1 score metrics.  

---

## Future Work
- Extend the model to handle multi-class land-cover classification.  
- Optimize preprocessing and augmentation for larger satellite datasets.  
- Explore deployment of trained models for real-time environmental monitoring.  

---
