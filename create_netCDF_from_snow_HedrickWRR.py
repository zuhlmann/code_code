import IPW_to_netCDF as ipw_nc
from awsm.data.topo import get_topo_stats
from smrf import ipw
import os

file_path_snow = '/mnt/snow/blizzard/tuolumne/devel/wy2016/wy2016_rerun_andrew_awsm_latedecay/runs/run20151001_20161001/output/'
file_name_out = 'WY2016_rerun_late_decay_snow2.nc'   #name of output nc file
file_path_out = '/home/zachuhlmann/tasks/scratch'  #this will be the file path where output file goes (Thanks Zach! I would never have figured that out.)
file_path_coords = '/home/zachuhlmann/tasks/zenodo_WRR_data/static_grids/tuolx_vegheight_50m.ipw'  # any file with coordinate/topo data from snow.ipw files

ipw_nc_obj = ipw_nc.IPW_to_netCDF(file_path_snow, file_path_out, file_name_out, file_path_coords)
ipw_nc_obj.get_files()
ipw_nc_obj.snow_nc()
