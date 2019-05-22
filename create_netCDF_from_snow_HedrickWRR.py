import IPW_to_netCDF as ipw_nc
# import matplotlib.pyplot as plt
from awsm.data.topo import get_topo_stats
from smrf import ipw
import os


fp_snow = '/mnt/snow/blizzard/tuolumne/devel/wy2013/wy2013_rerun_andrew_awsm_latedecay/runs/run20121001_20131001/output/'
file_list = os.listdir(fp_snow)

file_filt = []
for elem in file_list:
        if 'snow' in elem:
            file_filt.append(elem)
file_filt.sort()

file_num = []
for elem in file_filt:
    tmp = elem.split('.')
    file_num.append(int(tmp[1]))

#
# for i, j in enumerate(file_filt):
#     file_temp = fp_snow + j
#     print(file_temp)

fp_ipw = fp_snow + file_filt[100]



fp_mask = '/mnt/snow/blizzard/tuolumne/common_data/topo/tuolx_hetchy_mask_50m.ipw'
fp_dem = '/mnt/snow/blizzard/tuolumne/common_data/topo/tuolx_dem_50m.ipw'
fp_veg_type = '/mnt/snow/blizzard/tuolumne/common_data/topo/tuolx_vegnlcd_50m.ipw'
fp_veg_height = '/mnt/snow/blizzard/tuolumne/common_data/topo/tuolx_vegheight_50m.ipw'
fp_veg_k = '/mnt/snow/blizzard/tuolumne/common_data/topo/topo/tuolx_vegk_50m.ipw'
fp_veg_tau = '/mnt/snow/blizzard/tuolumne/common_data/topo/tuolx_vegtau_50m.ipw'

ts = get_topo_stats(fp_veg_height, filetype='ipw')  # collects all the coordinate data

##Add all the layers
# file_name = ['predicted_snow_depth', 'predicted_avg_snow_density', 'predicted_specific_mass', 'liquid_water_content',
#     'predicted_surface_temp', 'predicted_lower_layer_temp', 'predicted_avg_snow_temp', 'lower_layer_depth', 'percent_liquid_h20_saturation']

var_list = ['z_s', 'rho', 'm_s', 'h2o', 'T_s_0', 'T_s_l', 'T_s', 'z_s_l', 'h2o_sat']

var_data = {}
for i, j in enumerate(var_list):
    var_data[j] = ipw.IPW(fp_ipw).bands[i].data
var_data['x'] = ts['x']
var_data['y'] = ts['y']
# Not quite automated, but it will do

fp_out = '/home/zachuhlmann/projects/Hedrick_WRR_2018/snow.nc'
ipw_nc_obj = ipw_nc.IPW_to_netCDF(fp_out, var_data, var_list)
ipw_nc_obj.write_nc()
