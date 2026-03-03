import pandas as pd
import geopandas as gpd
#                  1.2)Filter satellite images whose center coordinates 
#                                  fall inside the region
# Load image 
images = pd.read_csv("../../data/image_coords.csv")  

# Convert to GeoDataFrame , lat , long degree style
gdf_images = gpd.GeoDataFrame(
    images,
    geometry=gpd.points_from_xy(images.longitude, images.latitude),
    crs="EPSG:4326"
)

# Load Delhi Airshed boundary
delhi_airshed = gpd.read_file("../../data/delhi_airshed.geojson")

# Spatial filter(filter which images are inside airshed)
filtered_images = gpd.sjoin(
    gdf_images,
    delhi_airshed,
    predicate="within",
    how="inner"
)
#                        1.3)Report the total number of images before and after filtering.
# Counts
print("Total images before filtering:", len(gdf_images))
print("Total images after filtering:", len(filtered_images))

# Drop geometry before saving
filtered_images_csv = filtered_images.drop(columns="geometry")

# Save filtered images
filtered_images_csv.to_csv("../../data/filtered_image_coords.csv", index=False)

print("Filtered images saved to data/filtered_image_coords.csv")