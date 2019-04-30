from __future__ import print_function
from osgeo import (ogr, gdal)
# # Import the "gdal" and "gdal_array" submodules from within the "osgeo" module
from osgeo import gdal_array
from osgeo import gdalconst
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# HOW TO CLIP FROM SHP
# gdalwarp -crop_to_cutline /home/zachuhlmann/projects/learning/landF/brb_shp /home/zachuhlmann/projects/learning/landF/lf64752427_US_140EVT/US_140EVT\US_140EVT.tif outup.tif
#
dataset = gdal.Open('/home/zachuhlmann/projects/learning/landF/landF_BRB_cropped.tif', gdal.GA_ReadOnly)
image_datatype = dataset.GetRasterBand(1).DataType
image = np.zeros((dataset.RasterYSize, dataset.RasterXSize),
  dtype = gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))
# Loop over all bands in dataset
band = dataset.GetRasterBand(1)
image[:,:] = band.ReadAsArray()

# if image needds clipping
# booli = np.where(image!=-9999)
# b1, b2 = min(booli[1]), max(booli[1])
# b3, b4 = min(booli[0]), max(booli[0])
#
# print(image.dtype)
# image = image.astype(np.float32)
# print(image.dtype)
# image[image==-9999]=np.nan
# f = plt.figure(num=0)
# a = plt.gca()
# a.imshow(image)
# plt.show()
# plt.imshow(image)

LF = pd.read_csv('LandFireSystems.csv', index_col = 'system code')

Keys_unique = np.unique(image)
Keys_unique = Keys_unique[~np.isnan(Keys_unique)]
Keys_unique = Keys_unique[~(Keys_unique==-9999)]
LFsub = LF.loc[Keys_unique.tolist()]
print(LFsub['system code'][0])


# # CREATE rat
# rat = gdal.RasterAttributeTable()
# print(rat)
# rat.CreateColumn("Key", gdalconst.GFT_Integer, gdalconst.GFU_MinMax)
# rat.CreateColumn("Name", gdalconst.GFT_String, gdalconst.GFU_Name)
# # rat.SetValueAsInt(0, 0, 0)
# rat.SetValueAsInt(0,0,17)
# rat.SetValueAsString(0,1,'zach')
# print(rat.GetRowCount())
# print(rat.GetValueAsString(0,1))







# GDALRasterAttributeTable::ValuesIO
