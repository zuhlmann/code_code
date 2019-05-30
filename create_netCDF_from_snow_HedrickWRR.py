import IPW_to_netCDFb as ipw_nc
# import matplotlib.pyplot as plt
from awsm.data.topo import get_topo_stats
from smrf import ipw
import os


fp_dem = '/mnt/snow/blizzard/tuolumne/common_data/topo/tuolx_dem_50m.ipw'
fp_snow = '/mnt/snow/blizzard/tuolumne/devel/wy2013/wy2013_rerun_andrew_awsm_latedecay/runs/run20121001_20131001/output/'
fd_out = '/home/zachuhlmann/projects/Hedrick_WRR_2018/'

ipw_nc_obj = ipw_nc.IPW_to_netCDF(fd_out, fp_snow, fp_dem)
ipw_nc_obj.get_files()
ipw_nc_obj.snow_nc()

# # ipw_nc_obj = ipw_nc.IPW_to_netCDF(fp_out, var_data)
# # ipw_nc_obj.write_nc()
