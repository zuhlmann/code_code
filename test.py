import getCDF as gcdf
import plotables as pltz
import matplotlib.pyplot as plt
import numpy as np
import h5py

from snowav.utils.MidpointNormalize import MidpointNormalize
import cmocean
import seaborn as sns
import matplotlib.colors as mcolors

import IPW_to_netCDF as ipw_nc

##########   USED WITH  awsm_test_case to include pds, di, hd, etc  #######
## RUN this for the pds, di, hd, etc: NOTE, may need work since GetCDF changed
# fn = '/home/zachuhlmann/code/code/snow.nc'
# gcdf_obj = gcdf.GetCDF(fn, "specific_mass", '2012-10-01 23:00:00', '2013-04-01 23:00:00', 'd')
# pds,di,hd = 6, 3, 4
# gcdf_obj.pull_times(pds, di, hd)
# img_str = '4_1_17_to_5_5_17_4hrs'
# gcdf_obj.plot_CDF(pds, di, hd, img_str)


########   WRR18 MAKING DIFF MAPS and adding functionality to IPW_to_netCDF   ########
# FILE PATHS
f_in = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow_WRR18_awsm_daily_cat2.nc'
var = 'specific_mass'
# Add mask to gcdf object
f_in_mask = '/home/zachuhlmann/projects/Hedrick_WRR_2018/tuol_topo_wrr18.nc'
f_in_hed = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow_WRR18_ipw_cat_zenodo.nc'

# NEW MODEL
gcdf_obj = gcdf.GetCDF()
gcdf_obj.init_nc_with_time(f_in, var, '2012-10-01 23:00:00', '2013-01-01 23:00:00', 'd')
gcdf_obj.get_topo(f_in_mask) #mask
# Necessary to init time indices
gcdf_obj.print_dates(10, 7)  # Keep either at 4 or multiples of 9 if saving plot. --> cont...
#Need to fix ~line 248 to only loop through number of times, not axs! ZRU 6/6/19
print('Number of observations: {0} \n April 1 index: {1}'.format(gcdf_obj.nobs, gcdf_obj.ids))

# OLD MODEL
gcdf_obj_hed = gcdf.GetCDF()
gcdf_obj_hed.init_nc_with_time(f_in_hed, var, '2012-10-01 23:00:00', '2013-01-01 23:00:00', 'd')

# # PLOT 3 PANEL: new, old, new - old
# gcdf_obj.plot_diff(gcdf_obj_hed)

# GET and PLOT Diff
gcdf_obj.get_diff(gcdf_obj_hed)
#
# #PLOT DIFF ONLY
# gcdf_obj.plot_diff_simple(2, 'delta_jan 3wk_test')
# #PRINT stats
# print('the basin diff is: ', gcdf_obj.acre_feet_delt_norm,  'acre feet norm')
# print('the basin area (masked) = {:.1f} acres' .format(gcdf_obj.basin_area))


# # # SAVE TO NC: save gdcf_obj.diff_mat[i *3 +2,:,:] to nc to visualize change
# fd_out = '/home/zachuhlmann/projects/Hedrick_WRR_2018'
# fp_dem = '/mnt/snow/blizzard/tuolumne/common_data/topo/tuolx_dem_50m.ipw'
# ipw_nc_obj = ipw_nc.IPW_to_netCDF(fd_out, fp_dem)
# ipw_nc_obj.mat_to_nc(gcdf_obj)

# Print table
print('')
print(gcdf_obj.df)

# gcdf_obj.plot_simple(dates)
