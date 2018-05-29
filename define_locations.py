# This script performs point-in-polygon matching for data entries to identify which of them are lying withing the
# Helsinki region
import pandas as pd
import geopandas as gpd
from shapely.geometry import *
import geojson
import csv

print("Importing Helsinki polygon data")
json_file = open('helsinki.json', 'r', encoding='utf-8')  # Reading the GeoJSON file
gson = geojson.loads(json_file.read())
gdf = gpd.GeoDataFrame.from_features(gson)  # Creating a data frame from the GeoJSON data
gdf.crs = {'init': 'epsg:3879'}  # Setting the source coordinate system
gdf.geometry = gdf.geometry.to_crs(epsg=4326)  # Converting to the traditional coordinate system
helsinki_poly = gdf.loc[1, 'geometry']  # Extracting the Helsinki region polygon
print("Importing Instagram data")
df = pd.read_csv('lang_instagram.tsv', sep='\t', encoding='latin-1')  # Reading the Instagram TSV
print("Performing point-in-polygon matching")
withinHelsinki = []
rows_count = float(len(df))
percentage = 0
print(str(percentage) + '%')
for index, row in df.iterrows():
    lat = float(row['lat'])
    lon = float(row['lon'])
    point = Point(lon, lat)
    withinHelsinki.append(helsinki_poly.intersects(point))  # Checking whether a point lies within the Helsinki polygon
    new_percentage = int(index/rows_count * 100)
    if new_percentage != percentage:
        percentage = new_percentage
        print(str(percentage) + '%')
df['within_Helsinki'] = pd.Series(withinHelsinki, index=df.index)  # Adding the within_Helsinki column to the data frame
print("Saving data to a file")
df.to_csv('loc_lang_instagram.tsv', sep='\t', quoting=csv.QUOTE_NONNUMERIC, index=False)
