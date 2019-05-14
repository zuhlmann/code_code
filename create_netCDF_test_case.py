import IPW_to_netCDF as ipw_nc
import matplotlib.pyplot as plt
from awsm.data.topo import get_topo_stats
from smrf import ipw

# file pointers (mostly)
fp_veg_height = '/home/zachuhlmann/projects/Test_Cases_GIS/data/height8.ipw'
veg_type_mat = ipw_nc.height_to_type(fp_veg_height)
fp_veg_tau = '/home/zachuhlmann/projects/Test_Cases_GIS/data/tau8.ipw'
fp_veg_k = '/home/zachuhlmann/projects/Test_Cases_GIS/data/mu8.ipw'  #Note: mu = k
fp_dem = '/home/zachuhlmann/projects/Test_Cases_GIS/data/dem.ipw'
fp_mask = '/home/zachuhlmann/projects/Test_Cases_GIS/data/mask.ipw'

ts = get_topo_stats(fp_veg_height, filetype='ipw')  # collects all the coordinate data

# Add all the layers
var_data = {}
var_data['veg_type'] = veg_type_mat
var_data['veg_height'] = ipw.IPW(fp_veg_height).bands[0].data
var_data['veg_tau'] = ipw.IPW(fp_veg_tau).bands[0].data
var_data['veg_k'] = ipw.IPW(fp_veg_k).bands[0].data
var_data['dem'] = ipw.IPW(fp_dem).bands[0].data
var_data['mask'] = ipw.IPW(fp_mask).bands[0].data
var_data['x'] = ts['x']
var_data['y'] = ts['y']
# Not quite automated, but it will do
var_list = ['veg_type', 'veg_height', 'veg_tau', 'veg_k', 'dem', 'mask']

fp_out = '/home/zachuhlmann/projects/Test_Cases_GIS/topo_RME2.nc'
ipw_nc_obj = ipw_nc.IPW_to_netCDF(fp_out, var_data, var_list)
ipw_nc_obj.write_nc()

# fig, ax = plt.subplots()
# im = ax.imshow(class_mat)
# plt.show()
