import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

import getCDF as gcdf


########   WRR18 MAKING DIFF MAPS and adding functionality to IPW_to_netCDF   ########
# FILE PATHS
f_in = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow.nc'
var = 'specific_mass'
# Add mask to gcdf object
f_in_mask = '/home/zachuhlmann/projects/Hedrick_WRR_2018/tuol_topo_wrr18.nc'
f_in_hed = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow_wrr18.nc'

# NEW MODEL
gcdf_obj = gcdf.GetCDF(f_in, var, '2012-10-01 23:00:00', '2013-01-01 23:00:00', 'd')

########   WRR18 MAKING DIFF MAPS and adding functionality to IPW_to_netCDF   ########
# FILE PATHS
f_in = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow.nc'
var = 'specific_mass'
# Add mask to gcdf object
f_in_mask = '/home/zachuhlmann/projects/Hedrick_WRR_2018/tuol_topo_wrr18.nc'
f_in_hed = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow_wrr18.nc'

# NEW MODEL
gcdf_obj = gcdf.GetCDF(f_in, var, '2012-10-01 23:00:00', '2013-01-01 23:00:00', 'd')
gcdf_obj.mask(f_in_mask) #mask

# Necessary to init time indices
gcdf_obj.print_dates(2, 7)
# print('Number of observations: {0} \n April 1 index: {1}'.format(gcdf_obj.nobs, gcdf_obj.ids))

# OLD MODEL
gcdf_obj_hed = gcdf.GetCDF(f_in_hed, var, '2012-10-01 23:00:00', '2013-01-01 23:00:00', 'd')
#
# GET and PLOT Diff
gcdf_obj.get_diff(gcdf_obj_hed)
time_list = []
[time_list.append(gcdf_obj.dt[idt]) for idt in gcdf_obj.idt]
df = pd.DataFrame({'time': time_list[:], 'basin_avg_new (mm)': gcdf_obj.avg_spec_mass_orig,
    'basin diff (acre_ft)': gcdf_obj.acre_feet_delt, '% change': gcdf_obj.acre_feet_delt_norm})
df = df.set_index('time')
print('')
print(df)
