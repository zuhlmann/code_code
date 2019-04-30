import os, sys
from osgeo import gdal

'''this bash script clips a tiff (from gdal_learn.py) using as inputs: west, east,
north and south extents'''

dset = gdal.Open(sys.argv[1])
west_ext, east_ext, north_ext, south_ext = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
tilesize = 30
gdaltransString = 'gdal_translate -of GTiff -srcwin '+str(west_ext)+', '+str(north_ext)+', '+str(east_ext-west_ext)+', ' \
    +str(south_ext-north_ext)+' '+sys.argv[1]+' landF_BRB_cropped.tif'
os.system(gdaltransString)

# gdaltranString = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
#             +str(h)+" " + sys.argv[1] + " " + sys.argv[2] + "_"+str(i)+"_"+str(j)+".tif"
#         os.system(gdaltranString)
