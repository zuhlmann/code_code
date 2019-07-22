import sys
# sys.path.insert(0, '/home/zachuhlmann/code/code')
import numpy as np
import rasterio as rio
import gdal_CL_utilities_v2 as gdalUtils
from snowav.utils.MidpointNormalize import MidpointNormalize
import matplotlib.pyplot as plt
import plotables as pltz
import time

# from pylab import *
# from scipy.optimize import curve_fit


fp_d1 = '/mnt/snowpack/lidar/Lakes/2019/USCALB20190501_SUPERsnow_depth_clipped.tif'
fp_d2 = '/mnt/snowpack/lidar/Lakes/2019/USCALB20190611_SUPERsnow_depth_clipped.tif'
fp_d1_2 = '/mnt/snowpack/lidar/SanJoaquin/2019/aso/USCASJ20190614_SUPERsnow_depth_50p0m_agg.tif'
fp_d2_2 = '/mnt/snowpack/lidar/SanJoaquin/2019/aso/USCASJ20190704_SUPERsnow_depth_50p0m_agg.tif'

fp_out = '/home/zachuhlmann/projects/basin_masks/'



utils_obj = gdalUtils.GDAL_python_synergy(fp_d1_2, fp_d2_2, fp_out)
utils_obj.clip_extent_overlap()
utils_obj.make_diff_mat()
# utils_obj.basic_stats()

# # utils_obj.save_tiff('mat_diff_norm')
# # utils_obj.replace_qml()

pltz_obj = pltz.Plotables()
pltz_obj.set_zero_colors(1)
pltz_obj.marks_colors()

# MAPPING OUTLIERS
name = ['mat_clip1', 'mat_diff_norm', 'mat_clip2']
operator = [['lt', 'gt'], ['lt', 'gt']]  #display options
val = [[17, -0.01], [10, -1.01]]   #display options
histogram_mats = ['mat_clip1', 'mat_diff_norm']
bin_dims = (60, 200)
moving_window_size = 3
moving_window_name = 'bins'
threshold_histogram_space = [0,-0.9,(4/9)]

# mov_wind(config['bin']['wind_sz'])
# mov_wind(config['map']['wind_sz'])

utils_obj.mask_advanced(name, operator, val)
utils_obj.hist_utils(histogram_mats, bin_dims)
# utils_obj.mov_wind(moving_window_name, moving_window_size)
start = time.time()
utils_obj.outliers_hist(threshold_histogram_space, moving_window_name, moving_window_size)  #fix this SOON!
end = time.time()
print('outliers hist time: ', end-start)
utils_obj.block_behavior(3, 0.20)
utils_obj.outliers_map()

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10,8))
asp_ratio = np.min(utils_obj.bins.shape) / np.max(utils_obj.bins.shape)
xedges, yedges = utils_obj.xedges, utils_obj.yedges

# Sub1: overall 2D hist
h = axes[0,0].imshow(utils_obj.bins, origin = 'lower', vmin=0.1, vmax = 8000, cmap = pltz_obj.cmap_choose,
     extent = (min(xedges), max(xedges), min(yedges), max(yedges)))
cbar = plt.colorbar(h, ax = axes[0,0])
cbar.set_label('bin count')
axes[0,0].title.set_text('2D histogram')
axes[0,0].set_xlabel('early date depth (m)')
axes[0,0].set_ylabel('relative delta snow depth')

# Sub2: clipped outliers
h = axes[0,1].imshow(utils_obj.outliers_hist_space, origin = 'lower',
    extent = (min(xedges), max(xedges), min(yedges), max(yedges)))
axes[0,1].title.set_text('outlier bins w/mov wind thresh: ' + str(round(threshold_histogram_space[2],2)))
axes[0,1].set_xlabel('early date depth (m)')
axes[0,1].set_ylabel('relative delta snow depth')

mat = utils_obj.trim_extent_nan('mat_diff_norm')
mat[~utils_obj.overlap_nan_trim] = np.nan

# Sub3: Basin snow map
h = axes[1,0].imshow(mat, origin = 'upper', cmap = pltz_obj.cmap_marks, norm = MidpointNormalize(midpoint = 0))
axes[1,0].title.set_text('First date snow depth')
cbar = plt.colorbar(h, ax = axes[1,0])
cbar.set_label('relative diff (%)')

# Sub4: Basin map of clipped snow
mat = utils_obj.trim_extent_nan('outliers_map_space')
mat[~utils_obj.overlap_nan_trim] = 0
h = axes[1,1].imshow(mat, origin = 'upper')
axes[1,1].title.set_text('locations of outliers (n=' + str(np.sum(utils_obj.outliers_map_space)) + ')')
axes[1,1].set_xlabel('snow depth (m)')
axes[1,1].set_ylabel('relative delta snow depth')
utils_obj.save_tiff('SJ_multiband', 'flag_gain_block')
# utils_obj.save_tiff('outliers_map_space', 'Lakes_06_11_05_01_outliers')

fig.suptitle('San Juoquin change 06/14 to 07/04')
# plt.savefig('/home/zachuhlmann/projects/basin_masks/SJ_07_04_06_14_2019_hist2d_outliers_wind_sz3_block.png', dpi=180)
plt.show()


# mat = utils_obj.trim_extent_nan('mat_diff_norm')
# mat[~utils_obj.overlap_nan_trim] = np.nan
# utils_obj.save_tiff(mat, 'mat_diff_norm_clip1')



# [hist, xbin, ybin] = np.histogram2d(x, y, bins = (50,50))
# fig, axes = plt.subplots(nrows = 2, ncols =1)
# utils_obj.mov_wind(hist, 5, 11/25)  # filters to pixels with 50% or more surrounding pixels present
# axes[0,0].imshow(utils_obj.mat_out, origin = 'lower', extent = (min(xbin), max(xbin), min(ybin), max(ybin)),
#     aspect = (max(xbin) - min(xbin)) / (max(ybin) - min(ybin)), vmax = 100)
# print('extent 1: ', (min(xbin), max(xbin), min(ybin), max(ybin)))
# axes[0,1].imshow(utils_obj.flag2, origin = 'lower', extent = (min(xbin), max(xbin), min(ybin), max(ybin)),
#     aspect = (max(xbin) - min(xbin)) / (max(ybin) - min(ybin)))
# xtrans, ytrans = xbin.copy(), ybin.copy()
# xtrans, ytrans = np.delete(xtrans, -1), np.delete(ytrans, -1)
# utils_obj.buff_points(1)
# # print(type(utils_obj.col_mx[0]))
# popt, pcoc = curve_fit(utils_obj.fractional_exp, xtrans[utils_obj.col_mx], ytrans[utils_obj.row_mx])
# xx = np.linspace(xtrans[0], xtrans[-1], 20)
# yy = utils_obj.fractional_exp(xx, *popt)
# # axes[1,0].plot(utils_obj.col_mx, utils_obj.row_mx, 'ko')
# axes[0,1].plot(xx,yy, 'r')
# axes[0,0].plot(xx,yy, 'r')
#
# plt.show()

# #1D HISTOGRAM
# mat = utils_obj.mat_diff_norm
# mat_nan = mat[~np.isnan(mat)]
# lb =utils_obj.lb
# ub = utils_obj.ub
# hist, bins = np.histogram(mat_nan, bins =30, range =(lb, ub))
# plt.bar(bins[:-1], hist, width = 0.1)
# plt.ylim(min(hist), 1.1 * max(hist))
# plt.show()
