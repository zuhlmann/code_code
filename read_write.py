import sys
sys.path.insert(0, '/home/zachuhlmann/code/code')
import plotables as pltz
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import gdal_CL_utilities as gdalUtils
import matplotlib.pyplot as plt
from pylab import *
from scipy.optimize import curve_fit
# import plotables as pltz


fp_d1 = '/mnt/snowpack/lidar/Lakes/2019/USCALB20190611_SUPERsnow_depth_clipped.tif'
fp_d2 = '/mnt/snowpack/lidar/Lakes/2019/USCALB20190501_SUPERsnow_depth_clipped.tif'
fp_d2_2 = '/mnt/snowpack/lidar/SanJoaquin/2019/aso/USCASJ20190614_SUPERsnow_depth_50p0m_agg.tif'
fp_d1_2 = '/mnt/snowpack/lidar/SanJoaquin/2019/aso/USCASJ20190704_SUPERsnow_depth_50p0m_agg.tif'
fn_out = '/home/zachuhlmann/projects/basin_masks/Lakes_hist2d_outliers_1000th.tif'

utils_obj = gdalUtils.GDAL_python_synergy(fp_d1, fp_d2, fn_out)
utils_obj.clip_min_extent()
utils_obj.diagnose()
# utils_obj.basic_stats()

# # utils_obj.save_tiff('mat_diff_norm')
# # utils_obj.replace_qml()

pltz_obj = pltz.Plotables()
pltz_obj.set_zero_colors(1)

# MAPPING OUTLIERS
name = ['mat_clip2', 'mat_diff_norm']
op = [['lt', 'gt'], ['lt', 'gt']]
val = [[17, 0.26], [10, -1.01]]
utils_obj.mask_advanced(name, op, val)
# utils_obj.hist_utils(['mat_clip2', 'mat_diff_norm'], (60, 200))
# utils_obj.thresh_hist(1000)  # change value for thresholding outliers
# utils_obj.map_flagged()

# fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10,8))
# asp_ratio = np.min(utils_obj.bins.shape) / np.max(utils_obj.bins.shape)
# xedges, yedges = utils_obj.xedges, utils_obj.yedges
#
# # Sub1: overall 2D hist
# h = axes[0,0].imshow(utils_obj.bins, origin = 'lower', vmin=0.1, cmap = pltz_obj.cmap_choose,
#      extent = (min(xedges), max(xedges), min(yedges), max(yedges)))
# cbar = plt.colorbar(h, ax = axes[0,0])
# cbar.set_label('bin count')
# axes[0,0].title.set_text('2D histogram')
# axes[0,0].set_xlabel('early date depth (m)')
# axes[0,0].set_ylabel('relative delta snow depth')
#
# # Sub2: clipped outliers
# h = axes[0,1].imshow(utils_obj.outliers, origin = 'lower',
#     extent = (min(xedges), max(xedges), min(yedges), max(yedges)))
# axes[0,1].title.set_text('clipped area')
# axes[0,1].set_xlabel('early date depth (m)')
# axes[0,1].set_ylabel('relative delta snow depth')
# mat = utils_obj.mat_clip2
# mat[~utils_obj.overlap] = np.nan
#
# # Sub3: Basin snow map
# h = axes[1,0].imshow(mat, vmax = 5, origin = 'upper')
# axes[1,0].title.set_text('First date snow depth')
# cbar = plt.colorbar(h, ax = axes[1,0])
# cbar.set_label('snow depth (m)')
#
# # Sub4: Basin map of clipped snow
# h = axes[1,1].imshow(utils_obj.hist_outliers, origin = 'upper',
#     extent = (min(xedges), max(xedges), min(yedges), max(yedges)))
# axes[1,1].title.set_text('locations of clipped areas')
# axes[1,1].set_xlabel('snow depth (m)')
# axes[1,1].set_ylabel('relative delta snow depth')
# utils_obj.save_tiff('hist_outliers')
#
# plt.savefig('Lakes_0611_0501_2019_Lakes_4panel_2dhist_clip.png', dpi=180)
# plt.show()





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
