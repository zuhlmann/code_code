import sys
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import gdal_CL_utilities as gdalUtils
import matplotlib.pyplot as plt


fp_d1 = '/mnt/snowpack/lidar/Lakes/2019/USCALB20190611_SUPERsnow_depth_clipped.tif'
fp_d2 = '/mnt/snowpack/lidar/Lakes/2019/USCALB20190501_SUPERsnow_depth_clipped.tif'
fn_out = '/home/zachuhlmann/projects/basin_masks/norm_diff.tif'

utils_obj = gdalUtils.GDAL_python_synergy(fp_d1, fp_d2, fn_out)
utils_obj.clip_min_extent()
utils_obj.diagnose(0.7)
utils_obj.save_tiff('mat_diff_norm')
# utils_obj.replace_qml()

mat = utils_obj.mat_diff_norm
mat_nan = mat[~np.isnan(mat)]
lb =utils_obj.lb
ub = utils_obj.ub
#
# # Histogram
# hist, bins = np.histogram(mat_nan, bins =30, range =(lb, ub))
# plt.bar(bins[:-1], hist, width = 0.1)
# plt.ylim(min(hist), 1.1 * max(hist))
# plt.show()

name = ['mat_clip2', 'mat_diff_norm']
op = [['lt', 'gt', 'fwd'], ['lt', 'fwd']]
val = [[5, 0.13, 2], [5, 2]]
mask = utils_obj.mask_advanced(name, op, val)
print(mask.shape)

x = utils_obj.mat_clip2[mask]
y = utils_obj.mat_diff_norm[mask]
print(np.max(x))

plt.hist2d(x,y, bins = (100,100))
plt.show()



# fig, axes = plt.subplots(nrows = 1, ncols =1)
# mp = axes.imshow(tmp)
# # mp.set_clim(utils_obj.lb, utilss_obj.ub)
# plt.show()
#
