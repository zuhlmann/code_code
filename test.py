import getCDF as gcdf
import plotables as pltz
import matplotlib.pyplot as plt
import numpy as np
import h5py


fn = '/home/zachuhlmann/code/code/snow.nc'
gcdf_obj = gcdf.GetCDF(fn, "specific_mass", '2012-10-01 23:00:00', '2013-04-01 23:00:00', 'd')
pds,di,hd = 6, 3, 4
gcdf_obj.pull_times(pds, di, hd)
img_str = '4_1_17_to_5_5_17_4hrs'
gcdf_obj.plot_CDF(pds, di, hd, img_str)
