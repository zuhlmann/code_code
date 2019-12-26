# import numpy as np
# import rasterio as rio
# import matplotlib.pyplot as plt
# from skimage import filters
#
# fp_snownc1 = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/run20190613_snow.nc'
# fp_snownc2 = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/run20190614_snow.nc'
#
# fp_snownc1_rio = 'netcdf:{}:{}'.format(fp_snownc1, 'thickness')
# with rio.open(fp_snownc1_rio) as src:
#     d1 = src
#     d1 = d1.read()
#     d1 = d1[0]
# fp_snownc2_rio = 'netcdf:{}:{}'.format(fp_snownc2, 'thickness')
# with rio.open(fp_snownc2_rio) as src:
#     d2 = src
#     d2 = d2.read()
#     d2 = d2[0]
#
# diff_arr = d1 - d2
# pct = np.sum(~np.isnan(diff_arr))
# num_outliers = 6000
# pct = (1 - (num_outliers / pct)) *100
# thresh = np.nanpercentile(diff_arr, pct)
# diff_arr[diff_arr < thresh] = thresh
#
# edges = filters.sobel(diff_arr)
#
# low = 0.1
# high = 0.35
#
# lowt = (edges > low).astype(int)
# hight = (edges > high).astype(int)
# hyst = filters.apply_hysteresis_threshold(edges, low, high)
#
# fig, ax = plt.subplots(nrows=2, ncols=2)
#
# ax[0, 0].imshow(diff_arr, cmap='gray')
# ax[0, 0].set_title('Original image')
#
# ax[0, 1].imshow(edges, cmap='magma')
# ax[0, 1].set_title('Sobel edges')
#
# ax[1, 0].imshow(lowt, cmap='magma')
# ax[1, 0].set_title('Low threshold')
#
# ax[1, 1].imshow(hight + hyst, cmap='magma')
# ax[1, 1].set_title('Hysteresis threshold')
#
# for a in ax.ravel():
#     a.axis('off')
#
# plt.tight_layout()
#
# plt.show()
####################
# fp = '/home/zachuhlmann/projects/data/SanJoaquin/SanJoaquin_topo.nc'
fp = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ20180601_SUPERsnow_depth.tif'
fp_clipped = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ20180601_SUPERsnow_depth_subsetZ.tif'

import basin_setup.basin_setup as bs
import copy
import math
from subprocess import run
ext = bs.parse_extent(fp, cellsize_return = True)
xmin, ymin, xmax, ymax = ext[0], ext[1], ext[2], ext[3]
x_meters = xmax - xmin
y_meters = ymax - ymin
box_options = []
subset_pct = 0.25
print(ext)
#
# # make a box subset using x dimension of image as basis for size
# # approximate subset_box size
# subset_box_length = math.ceil(x_meters * subset_pct)
# print(subset_box_length)
# rem = subset_box_length % 50
# print(rem)
# # get subset box dim to multiple of spatial rez
# subset_box_length = subset_box_length - rem
# print(subset_box_length)
# # ensure that subset box length is divisible by both spatial resolutions
# # start from original subset_box length and grow by multiples of smaller rez
# # (3) until divisible by both large (50) and small (3) rez
# for i in range(0,1000,50):
#     div_by_3 = ((subset_box_length + i) % 3) == 0
#     div_by_50 = ((subset_box_length + i) % 50) == 0
#     if not (div_by_3 and div_by_50):
#         pass
#     else:
#         subset_box_length = subset_box_length + i
#         break
#
# # get first x offset
# off_pct = (1 - subset_pct) / 2
# x_off1 = x_meters * off_pct
# rem = x_off1 % 3
# # round to whole number divisible by smaller rez (3)
# x_off1 = x_off1 - rem
# x_off_idx1 = x_off1 + xmin
# x_off_idx2 = x_off_idx1 + subset_box_length
#
#
# # get first x offset
# y_off1 = y_meters * off_pct
# rem = y_off1 % 3
# # round to whole number divisible by smaller rez (3)
# y_off1 = y_off1 - rem
# y_off_idx1 = y_off1 + ymin
#
# y_off_idx2 = y_off_idx1 + subset_box_length
# off_idx = [x_off_idx1, y_off_idx1, x_off_idx2, y_off_idx2]
# gdal_str = []
# [gdal_str.append(int(te)) for te in off_idx]
# gdal_str = 'gdalwarp -te {} {} {} {} {} {} -overwrite'.format(*gdal_str, fp, fp_clipped)
# print('\n', gdal_str, '\n')
# run(gdal_str, shell = True)



############################
# import rasterio as rio
# fp = '/home/zachuhlmann/projects/data/SanJoaquin/SanJoaquin_topo.nc'
# fp_rio = 'netcdf:{}:mask'.format(fp)
# with rio.open(fp_rio) as src:
#     meta = src.profile
#
# rez = meta['transform'][0]
# print(rez ** 2)

#################################
# pct_list = [0.1, 0.2, 0.3]
# pct_list = ['{}% thresh'.format(int(pct * 100)) for pct in pct_list]
# print(pct_list)

####################################
# import json
# file_path_out_json = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/USCASJ20190614_20190704_metadata.txt'
#
# with open(file_path_out_json) as meta_orig:
#     meta_orig = json.load(meta_orig)
# # b) add basin statistics to dictionary
# gaining = True
# basin_total_change = 2000
# basin_avg_change = 1.5
#
# meta_orig.update({'gaining':gaining,
#                 'basin_total_change':basin_total_change,
#                 'basin_avg_change':basin_avg_change})
# # c) write updated metadata to file
# with open(file_path_out_json, 'w') as meta_plus_basin_data:
#     json.dump(meta_orig, meta_plus_basin_data)

############################################
# fp_tif = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ20190614_SUPERsnow_depth_50p0m_agg.tif'
# fp_nc = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/run20190614_snow.nc'
# fp_nc_out = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/run20190614_snow_clip.nc'
# fp_nc_rio = 'netcdf:{0}:{1}'.format(fp_nc, 'thickness')
# fp_tif_clipped_nc = '/home/zachuhlmann/projects/data/SanJoaquin/20190703_20190704/USCASJ20190704_clipped_to_run20190703_snownc.tif'
# fp_tif_orig = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ20190704_SUPERsnow_depth_50p0m_agg_merged.tif'
#
#
# import basin_setup.basin_setup as bs
# import rasterio as rio
# from subprocess import run
# from netCDF4 import Dataset
#
# ext_tif = bs.parse_extent(fp_tif_clipped_nc, cellsize_return = True)
# ext_nc = bs.parse_extent(fp_nc, cellsize_return = True)
#
# with rio.open(fp_tif_clipped_nc) as src:
#     d1 = src
#     meta = src.profile
# with rio.open(fp_tif_orig) as src:
#     d1 = src
#     meta2 = src.profile
#
# print('meta orig:\n', meta2)
# print('meta clip:\n', meta)
#
# print(ext_tif)
# xdims = (ext_tif[2] - ext_tif[0]) / ext_tif[4]
# ydims = (ext_tif[3] - ext_tif[1]) / ext_tif[4]
# print('\n')
# print('dims: ', xdims, ydims)
# print(d1.bounds)
#
#  [259392.0, 4096625.0, 353325.0, 4179575.0]
#########################################
# import datetime
# year = 2019
# month = 10
# day1 = 20
# day2 = 21
# datetime_obj = datetime.date(year, month, day1)
# datetime_obj2 = datetime.date(year, month, day2)
# diff = datetime_obj2 - datetime_obj
# one_day = datetime.timedelta(days = 1)
# print(diff==one_day)



# min_extents = ext_nc[:4]
# min_ext_temp = []
# [min_ext_temp.append(ext + 50 * 200) for ext in min_extents[:2]]
# [min_ext_temp.append(ext - 50 * 200) for ext in min_extents[2:4]]
#
# tr_substring = '-of NETCDF NETCDF:"{0}":thickness {1}'.format(fp_nc, fp_nc_out)
# #
#
# run_arg3 = 'gdalwarp -te {0} {1} {2} {3} {4} -overwrite'.format \
#             (*min_ext_temp, tr_substring)
# print(run_arg3)
# run(run_arg3, shell = True)
