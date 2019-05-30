import pandas as pd
import numpy as np
from smrf import ipw
import netCDF4 as nc
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num
from spatialnc.proj import add_proj
from awsm.data.topo import get_topo_stats
import os
import smrf.utils as utils

# Note: adapted from Micah Johnson script which was adapted from Basin_Setup
# ZRU 5/23/19:  Earlier Version, [not <IPW...CDFb>] needs to be incorporated into this.
# - with a def topo_nc() and an option in base_nc() for topo or snow to determine if time
# will be added as dimension or not.
class IPW_to_netCDF():
    def __init__(self, fd_out, fp_name, fp_dem):
        self.fd_out = fd_out
        self.fp_dem = fp_dem
        self.fp_name = fp_name

    def snow_nc(self):
        # S is a dictionary list or something which is hardcoded band names and attributes for netCDF
        # from AWSM/convertfiles/convertFiles
        s = {}
        s['name'] = ['thickness', 'snow_density', 'specific_mass', 'liquid_water',
                     'temp_surf', 'temp_lower', 'temp_snowcover',
                     'thickness_lower', 'water_saturation']
        s['units'] = ['m', 'kg m-3', 'kg m-2', 'kg m-2',
                      'C', 'C', 'C', 'm', 'percent']
        s['description'] = ['Predicted thickness of the snowcover',
                            'Predicted average snow density',
                            'Predicted specific mass of the snowcover',
                            'Predicted mass of liquid water in the snoipw.IPW(v2).bands[i].datawcover',
                            'Predicted temperature of the surface layer',
                            'Predicted temperature of the lower layer',
                            'Predicted temperature of the snowcover',
                            'Predicted thickness of the lower layer',
                            'Predicted percentage of liquid water']

        # Tell file where to save and name
        self.fp_out = os.path.join(self.fd_out, 'snow_WRR18_2day.nc')  #fd = file directory, fp = file path
        # Get X,Y and Time saved to nc file
        snow = self.base_nc()
        for i, v in enumerate(self.fp_ipw):
            snow.variables['time'][i] = self.file_num[i]    #file number is actually the hours after WY i.e. 23, 47, etc.
            for i2, v2 in enumerate(s['name']):
                # snow.createVariable(v, 'f', self.dimensions[:3], chunksizes=(6, 10, 10))
                if i == 0:
                    snow.createVariable(v2, 'f', self.dimensions[:3])
                    setattr(snow.variables[v2], 'units', s['units'][i2])
                    setattr(snow.variables[v2], 'description', s['description'][i2])
                    snow.variables[v2][i,:,:] = ipw.IPW(v).bands[i2].data
                else:
                    snow.variables[v2][i,:,:] = ipw.IPW(v).bands[i2].data
                    # snow.variables[v][0,:,:] = self.var_data[v]
                    # print(snow.variables[v][0,:,:].shape)
                    # print(self.var_data[v].shape)

        h = '[{}] Data added or updated'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        snow.setncattr_string('last modified', h)
        snow.setncattr_string('AWSM version', '9.15')  #FIX THIS!  as below line commented
        # snow.setncattr_string('AWSM version', 'myawsm.gitVersion')

        snow.setncattr_string('Conventions', 'CF-1.6')
        snow.setncattr_string('institution',
                'USDA Agricultural Research Service, Northwest Watershed Research Center')

        snow.setncattr_string('last_modified', h)
        snow = add_proj(snow, 26711)
        snow.sync()
        snow.close()

    def base_nc(self):
        # this begins netCDF file with x, y and time.  to be used in conjunction with topo or snow
        # to add respective data and close file in snow_nc() for example
        base_nc = nc.Dataset(self.fp_out, 'w')
        ts = get_topo_stats(self.fp_dem, filetype='ipw')  # collects all the coordinate data
        x = ts['x']
        y = ts['y']

        hours_raw = self.file_num
        dimensions = ('time', 'y', 'x')
        base_nc.createDimension(dimensions[0], None)
        base_nc.createDimension(dimensions[1], y.shape[0])
        base_nc.createDimension(dimensions[2], x.shape[0])

        base_nc.createVariable('time', 'f', dimensions[0])
        base_nc.createVariable('y', 'f', dimensions[1])
        base_nc.createVariable('x', 'f', dimensions[2])

        setattr(base_nc.variables['y'], 'units', 'meters')
        setattr(base_nc.variables['y'], 'description', 'UTM, north south')
        setattr(base_nc.variables['y'], 'long name', 'y coordinate')
        setattr(base_nc.variables['y'], 'standard name', 'projection_y_coordinate')
        # ALTERNATE WAY TO setattr()
        #s.variables['y'].setncattr(
                # 'units',
                # 'meters')
        setattr(base_nc.variables['x'], 'units', 'meters')
        setattr(base_nc.variables['x'], 'description', 'UTM, east west')
        setattr(base_nc.variables['x'], 'long name', 'x coordinate')
        setattr(base_nc.variables['x'], 'standard name', 'projection_x_coordinate')
        # setattr(base_nc.variables['time'], 'units',
        #         'hours since %s' % 'myawsm.wy_start')
        #fix to function as above comment
        setattr(base_nc.variables['time'], 'units', 'hours since 2012-10-01 00:00:00')
        setattr(base_nc.variables['time'], 'calendar', 'standard')
        setattr(base_nc.variables['time'], 'time_zone', 'UTC')
        # consider putting time calculation into function or check awsm/convertFiles/convertFiles (plus one more?) for functions
        time_w = []
        dates = [datetime(2012,10,1) + hours_raw * timedelta(hours = 1) for  hours_raw in hours_raw]
        time_w[:] = date2num(dates, units = base_nc['time'].units, calendar = base_nc['time'].calendar)
        base_nc.variables['time'][:] = time_w
        base_nc.variables['x'][:] = x
        base_nc.variables['y'][:] = y
        self.dimensions = dimensions   # Prob more elegant way to do this
        return base_nc

    def get_files(self):
        # grabs all files from input path
        file_list = os.listdir(self.fp_name)

        # parse for snow.nc
        file_filt = []
        for elem in file_list:
                if 'snow' in elem:
                    file_filt.append(elem)
        file_filt.sort()

        # number will be hours since start date (Oct 1)
        file_num = []
        for ct, elem in enumerate(file_filt):
            tmp = elem.split('.')
            file_num.append(int(tmp[1]))

        # create filepath to file
        fp_ipw = [None] * len(file_filt)
        for i in range(len(file_filt)):
            fp_ipw[i] = self.fp_name + file_filt[i]

        self.file_num = file_num
        self.fp_ipw = fp_ipw

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


# IN CASE looking for location of topo files from original usage of this script
# fp_veg_height = '/home/zachuhlmann/projects/Test_Cases_GIS/data/height8.ipw'
