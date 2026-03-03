#import dependencies
import os
import rasterio
from rasterio.windows import Window
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torchvision.transforms as T
import torchvision.models as models
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

#paths of rgb data ,image cor-ord,tif path
RGB_DIR = "../data/rgb/"
ESA_PATH = "../data/worldcover_bbox_delhi_ncr_2021.tif"
CSV_PATH = "../data/filtered_image_coords.csv"

#open .tif file
esa_src = rasterio.open(ESA_PATH)
#                   2.1)For each image, extract the  128×128 corresponding land-cover 
#                         patch from land_cover.tif using its center coordinate 

def extract_esa_patch(lat, lon, size=128):
    row, col = esa_src.index(lon, lat)
    half = size // 2
    window = Window(col - half, row - half, size, size)
    return esa_src.read(1, window=window)
#                        2.2)Assign the image label using the dominant (mode) land-cover class.

def dominant_landcover(patch):
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return int(mode(patch, axis=None, keepdims=False).mode)

# Load CSV and find dominant landcover for each row
df = pd.read_csv(CSV_PATH)

df["esa_class"] = df.apply(
    lambda r: dominant_landcover(extract_esa_patch(r.latitude, r.longitude)),
    axis=1
)
#               2.3)Map ESA class codes to simplified land-use categories 
#                (e.g., Built-up, Vegetation, Water, Cropland, Others).
esa_to_landuse = {
    10: "Vegetation",
    20: "Vegetation",
    30: "Vegetation",
    40: "Cropland",
    50: "Built-up",
    80: "Water"
}

df["landuse"] = df["esa_class"].map(esa_to_landuse)
#drop null values
df = df.dropna(subset=["landuse"])
#                  2.4)Perform a 60/40 train-test split randomly and visualize class distribution
df = df.sample(n=800, random_state=42).reset_index(drop=True)
print("Using samples:", len(df))
train_df, test_df = train_test_split(
    df, test_size=0.4, random_state=42, shuffle=True
)
print("Train size:", len(train_df))
print("Test size:", len(test_df))
#get sorted names of classes
classes = sorted(df["landuse"].unique())
#label them
label_map = {c: i for i, c in enumerate(classes)}

# Image mapping

class LandUseDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
#fetch length of input dataset
    def __len__(self):
        return len(self.df)
#get the rgb image fromm csv and label with its landuse
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(RGB_DIR, row["filename"])
        print("Loading:", img_path)   
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = label_map[row.landuse]
        return image, label

train_ds = LandUseDataset(train_df)
test_ds = LandUseDataset(test_df)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=0)


#                             3.1)BUILD AND TRAIN A MODEL(ResNet18)
#gpu try if not present use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#use pretrained resnet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training

epochs = 10
for e in range(epochs):
    model.train()
    loss_sum = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {e+1}/{epochs}, Loss: {loss_sum:.4f}")

#                            3.2)Evaluate using accuracy and F1-score

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1)
        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")

print(f"Accuracy: {acc*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")

#                           3.3)Display a confusion matrix and briefly interpret the results

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=classes,
            yticklabels=classes,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
#save the model(extra)
MODEL_PATH = "landuse_resnet18.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": classes
}, MODEL_PATH)

print(f"Model saved at {MODEL_PATH}")