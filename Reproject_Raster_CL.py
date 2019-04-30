import sys, os
from osgeo import gdal
# NOte originally used to transform a .e00 veg file (argGIS) ZRU 4/2019

in_file = sys.argv[1]
s_srs = sys.argv[2]
t_srs = sys.argv[3]
gdalwarpString = 'gdalwarp -s_srs '+str(s_srs)+' -t_srs '+str(t_srs)+' -of netCDF -dstnodata [6] -overwrite '+str(in_file)+'  veg_WGS84_UTM11_v2.nc'
print(gdalwarpString)
os.system(gdalwarpString)
