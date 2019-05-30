import getCDF as gcdf
import plotables as pltz
import matplotlib.pyplot as plt
import numpy as np
import h5py

from snowav.utils.MidpointNormalize import MidpointNormalize
import cmocean
import seaborn as sns
import matplotlib.colors as mcolors
#
## RUN this for the pds, di, hd, etc: NOTE, may need work since GetCDF changed
# fn = '/home/zachuhlmann/code/code/snow.nc'
# gcdf_obj = gcdf.GetCDF(fn, "specific_mass", '2012-10-01 23:00:00', '2013-04-01 23:00:00', 'd')
# pds,di,hd = 6, 3, 4
# gcdf_obj.pull_times(pds, di, hd)
# img_str = '4_1_17_to_5_5_17_4hrs'
# gcdf_obj.plot_CDF(pds, di, hd, img_str)

# FILE PATHS
f_in = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow.nc'
var = 'specific_mass'
# Add mask to gcdf object
f_in_mask = '/home/zachuhlmann/projects/Hedrick_WRR_2018/tuol_topo_wrr18.nc'
f_in_hed = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow_WRR18_2day.nc'

# NEW MODEL
gcdf_obj = gcdf.GetCDF(f_in, var, '2012-10-01 23:00:00', '2013-01-01 23:00:00', 'd')
gcdf_obj.mask(f_in_mask) #mask
# Necessary to init time indices
gcdf_obj.print_dates(9, 15)
print('Number of observations: {0} \n April 1 index: {1}'.format(gcdf_obj.nobs, gcdf_obj.ids))

# OLD MODEL
gcdf_obj_hed = gcdf.GetCDF(f_in_hed, var, '2012-10-01 23:00:00', '2013-01-01 23:00:00', 'd')

# # PLOT 3 panel: new, old, new - old
# gcdf_obj.plot_diff(gcdf_obj_hed)

# # GET and PLOT Diff
gcdf_obj.get_diff(gcdf_obj_hed)
gcdf_obj.plot_diff_simple(2, 'delta_jan to march_1_5_b')
# PRINT stats
print('the basin diff is: ', gcdf_obj.acre_feet_delt,  'acre feet')
print('the basin area (masked) = {:.1f} acres' .format(gcdf_obj.basin_area))





# gcdf_obj.plot_simple(dates)
