import pandas as pd
import numpy as np
from smrf import ipw
import netCDF4 as nc
from datetime import datetime
from spatialnc.proj import add_proj
# Note: adapted from Micah Johnson script which was adapted from Basin_Setup

class IPW_to_netCDF():
    def __init__(self, fp_out, var_data, var_list):
        self.fp_out = fp_out
        self.var_data = var_data
        self.var_list = var_list

    def write_nc(self):
        s = nc.Dataset(self.fp_out, 'w',
                       format='NETCDF4', clobber=False)

        x = self.var_data['x']
        y = self.var_data['y']
        dimensions = ('y', 'x')

        s.createDimension(dimensions[0], y.shape[0])
        s.createDimension(dimensions[1], x.shape[0])

        # create the variables
        s.createVariable('y', 'f', dimensions[0])
        s.createVariable('x', 'f', dimensions[1])

        s.variables['y'].setncattr(
                'units',
                'meters')
        s.variables['y'].setncattr(
                'description',
                'UTM, north south')
        s.variables['y'].setncattr(
                'long_name',
                'y coordinate')
        s.variables['y'].setncattr(
                'standard_name',
                'projection_y_coordinate')
        # the x variable attributes
        s.variables['x'].setncattr(
                'units',
                'meters')
        s.variables['x'].setncattr(
                'description',
                'UTM, east west')
        s.variables['x'].setncattr(
                'long_name',
                'x coordinate')
        s.variables['x'].setncattr(
                'standard_name',
                'projection_x_coordinate')
        s.variables['y'][:] = y
        s.variables['x'][:] = x

        for idx, vr in enumerate(self.var_list):
            type_list = ['u1', 'f', 'f', 'f', 'f', 'u1']
            long_name = ['vegetation type', 'vegetation height', 'vegetation tau', 'vegetation k', 'dem', 'mask']
            s.createVariable(vr, type_list[idx],
                             (dimensions[0], dimensions[1]))

            # the variable attributes
            s.variables[vr].setncattr(
                    'long_name',
                    long_name[idx])
            s.variables[vr].setncattr(
                    'grid_mapping',
                    'projection')

            s.variables[vr][:] = self.var_data[vr]
        s.setncattr_string('Conventions', 'CF-1.6')
        s.setncattr_string('institution',
                'USDA Agricultural Research Service, Northwest Watershed Research Center')
        h = '[{}] Data added or updated'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        s.setncattr_string('last_modified', h)
        s = add_proj(s, 26711)
        s.sync()
        s.close()

def height_to_type(veg_height):
    '''ZRU: this takes veg_height.ipw, reclassifies as 1,2,...N (N = number of distinct veg heights),
    and outputs nparray.  Note: classifications are ordered from shortest to tallest heights'''
    mat = ipw.IPW(veg_height).bands[0].data  # grab veg_height array from ipw image
    hts=[]   #init list to collect unique heights
    for i in np.arange(mat.shape[0]):
        for j in np.arange(mat.shape[1]):
            if mat[i,j] in hts:
                pass
            else:
                hts.append(mat[i,j])
    cls = mat.copy()
    hts.sort()
    for idx, vals in enumerate(hts):
        cls[cls==vals] = idx+1
    return cls

# # file pointers (mostly)
# fp_veg_height = '/home/zachuhlmann/projects/Test_Cases_GIS/data/height8.ipw'
# veg_type_mat = height_to_type(fp_veg_height)
# fp_veg_tau = '/home/zachuhlmann/projects/Test_Cases_GIS/data/tau8.ipw'
# fp_veg_k = '/home/zachuhlmann/projects/Test_Cases_GIS/data/mu8.ipw'  #Note: mu = k
# fp_dem = '/home/zachuhlmann/projects/Test_Cases_GIS/data/dem.ipw'
# fp_mask = '/home/zachuhlmann/projects/Test_Cases_GIS/data/mask.ipw'
#
# ts = get_topo_stats(fp_veg_height, filetype='ipw')  # collects all the coordinate data
#
# # Add all the layers
# var_data = {}
# var_data['veg_type'] = veg_type_mat
# var_data['veg_height'] = ipw.IPW(fp_veg_height).bands[0].data
# var_data['veg_tau'] = ipw.IPW(fp_veg_tau).bands[0].data
# var_data['veg_k'] = ipw.IPW(fp_veg_k).bands[0].data
# var_data['dem'] = ipw.IPW(fp_dem).bands[0].data
# var_data['mask'] = ipw.IPW(fp_mask).bands[0].data
# var_data['x'] = ts['x']
# var_data['y'] = ts['y']
# # Not quite automated, but it will do
# var_list = ['veg_type', 'veg_height', 'veg_tau', 'veg_k', 'dem', 'mask']
#
# fp_out = '/home/zachuhlmann/projects/Test_Cases_GIS/topo_RME.nc'
# write_nc(fp_out, var_data, ts, var_list)
