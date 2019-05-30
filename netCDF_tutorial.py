# FROM THIS WEBSITE:
# http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/11_create_netcdf_python.pdf

import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num

dset = Dataset('test.nc', 'w')
print(dset.file_format)

# CREATING DIMENSIONS
#  dimensions are actually variables, referred to as 'Coordinate Variaibles'
level = dset.createDimension('level', 10)
lat = dset.createDimension('lat',73)
lon = dset.createDimension('lon', 144)
time = dset.createDimension('time', None)

#  CREATING VARIABLES
# Notice formate  var = dset.createVariable(<name>, <datatype>, <dimension name or number (if no dimensin created))
times = dset.createVariable('time', np.float64, ('time',))
levels = dset.createVariable('level', np.int32, ('level',))  #I believe level is a proxy for some variable (i.e. rH)
latitudes = dset.createVariable('latitude', np.float32, ('lat',))
longitudes = dset.createVariable('longitude', np.float32, ('lon',))
temp = dset.createVariable('temp', np.float32, ('time','level','lat','lon'))

# ACCESSING VARS
print('temp variable:', dset.variables['temp'])

#  ATTRIBUTES GLOBAL
import time
dset.desciption = 'bogus example script'
dset.history = 'Created' + time.ctime(time.time())
dset.source = 'netCDF4 python module tutorial'

# ATTRIBUTE VARIABLES
latitudes.units = 'degree_north'
longitudes.units = 'degree_east'
levels.units = 'hPa'
temp.units = 'K'

times.units = 'hours since 0001-01-01 00:00:00'
times.calendar = 'gregorian'

# WRITING Dataset
lats = np.arange(-90,91,2.5)  #notice, this is 73 elements
lons = np.arange(-180,180,2.5)  #notice, this is 144 elements
latitudes[:] = lats
longitudes[:] = lons

#GROWING DATA ALONG UNLIMITED DIMENSION
print('temp shape before adding data = ', temp.shape)
from numpy.random import uniform
nlats = len(dset.dimensions['lat'])
nlons = len(dset.dimensions['lon'])
temp[0:5,0:10,:,:] = uniform(size=(5,10,nlats,nlons))   #added five time dimensions(?); see! 0:5,
print('temp shape after adding data: ', temp.shape)

# DEFINING DATE/TIMES CORRECTLY
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num
dates = []


for n in range(temp.shape[0]):
    dates.append(datetime(2001,3,1) + n * timedelta(hours=12))
times[:] = date2num(dates,units = times.units, calendar = times.calendar)
print('time values (in units %s: ' % times.units + '\n', times[:])
dates = num2date(times[:], units=times.units, calendar=times.calendar)
print('dates corresponding to time values:\n', dates)

# Dates: section 7)
date = [datetime(1984,10,18)+n*timedelta(hours=12) for n in range(temp.shape[0])]
print(date[2])
print([n for n in range(temp.shape[0])])
# WRITE THE FILE
dset.close()
