#install dependencies
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import numpy as np
#read geojsonfile
delhi_ncr = gpd.read_file("../../data/delhi_ncr_region.geojson")
#cover lat long from degree to metres using  espg
delhi_ncr_utm = delhi_ncr.to_crs(epsg=32644)
#boundaries of delhi ncr
minx, miny, maxx, maxy = delhi_ncr_utm.total_bounds
#creare 60km*60km grid
grid_size = 60000  
#define grid cells
grid_cells = []
x_coords = np.arange(minx, maxx, grid_size)
y_coords = np.arange(miny, maxy, grid_size)
#create box for each grid
for x in x_coords:
    for y in y_coords:
        grid_cells.append(box(x, y, x + grid_size, y + grid_size))
#convert grid to geojson form
grid = gpd.GeoDataFrame(geometry=grid_cells, crs=delhi_ncr_utm.crs)

fig, ax = plt.subplots(figsize=(8, 8))
#plot delhi-ncr map
delhi_ncr_utm.plot(ax=ax, color='none', edgecolor='black')
#plot grid
grid.plot(ax=ax, color='none', edgecolor='green', linewidth=0.2)
plt.title("Delhi-NCR with 60×60 km Grid")
plt.show()