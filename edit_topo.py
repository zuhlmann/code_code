import netCDF4 as nc
import numpy as np

filename = '/home/zachuhlmann/projects/Test_Cases_GIS/topo.nc'
ncfile = nc.Dataset(filename,'a')
lu_index = ncfile.variables['veg_type'][:]
I = np.where(lu_index == 3139)
lu_index[I] = 3
ncfile.variables['veg_type'][:] = lu_index
ncfile.close()

print('conversion complete')
