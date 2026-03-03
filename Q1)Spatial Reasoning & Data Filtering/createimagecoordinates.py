import os
import pandas as pd

image_dir = "data/rgb"

data = []

for fname in os.listdir(image_dir):
    if fname.endswith(".png"):
        name = fname.replace(".png", "")
        
        # EXAMPLE FORMAT: img_28.6139_77.2090.png
        parts = name.split("_")
        
        lat = float(parts[-2])
        lon = float(parts[-1])
        
        data.append({
            "filename": fname,
            "latitude": lat,
            "longitude": lon
        })

df = pd.DataFrame(data)
df.to_csv("image_coords.csv", index=False)

print("image_coords.csv created with", len(df), "entries")
