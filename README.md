# Land-Cover Classification and ResNet18-Based Image Analysis

## Overview
This project focuses on classifying satellite images using deep learning techniques, specifically the **ResNet18** model. The main goal is to assign image labels based on the **dominant (mode) land-cover class** and explore AI applications for environmental and societal problem-solving.  

The project demonstrates the use of geospatial data processing, image analysis, and deep learning to achieve high-accuracy land-cover classification.

---

## Features
- Assign labels to satellite images using the dominant (mode) land-cover class.  
- Filter and preprocess satellite imagery for analysis.  
- Implement a **ResNet18** convolutional neural network for image classification.  
- Evaluate model performance using **accuracy** and **F1 score** metrics.  

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

## Acknowledgements
I acknowledge that I have taken the help of the AI tool **ChatGPT** to implement some tasks, such as assigning image labels using the dominant (mode) land-cover class, filtering satellite images, and implementing the ResNet18 model. However, I have a **conceptual understanding** of how the code works and the purpose of the libraries, methods, and functions used.  

---

## How to Run
1. Clone this repository.  
2. Install the required libraries listed above using `pip install`.  
3. Prepare the dataset (satellite imagery) in the designated folder.  
4. Run the main script to preprocess images, assign labels, and train the ResNet18 model.  
5. Evaluate the model using accuracy and F1 score metrics.  

---

## Future Work
- Extend the model to handle multi-class land-cover classification.  
- Optimize preprocessing and augmentation for larger satellite datasets.  
- Explore deployment of trained models for real-time environmental monitoring.  

---

