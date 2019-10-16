import rasterio as rio
# from rasterio.rio.info import info
from matplotlib import pyplot as plt
# from memory_profiler import profile
import json
import affine

# file_path_temp = '/mnt/snowpack/lidar/SanJoaquin/2019/USCASJ20190704_SUPERsnow_depth_50p0m_agg_merged.tif'
file_path_temp = 'netcdf:/home/zachuhlmann/projects/data/SanJoaquin_2019_topo.nc:dem'
#
with rio.open(file_path_temp) as src:
    d1 = src
    meta = d1.profile

print(meta)

# same_rez = abs(meta['transform'][0]) == abs(meta['transform'][4])
# print(meta)
# print(rio.crs.CRS.from_epsg(32611) == meta['crs'])
# with open('test_json.txt', 'w') as outfile:
#     data = json.dumps(dict({k:v for k, v in meta.items() if k != 'crs'}))
#     data2 = dict({k:v for k, v in meta.items() if k != 'crs'})
#     data2.update({'crs' : 32611})
#     json.dump(data2, outfile)
#
# with open('test_json.txt') as balls:
#     temp = json.load(balls)
#
# temp.update({'crs' : rio.crs.CRS.from_epsg(temp['crs'])})
# print(temp)
#
# crs_obj = meta['crs']
# print(type(rio.crs.CRS.to_epsg(crs_obj)))
