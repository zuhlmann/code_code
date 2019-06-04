import h5py
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# maxus.nc for Hedrick_WRR18 had coordinate issues.  Fixed.

#  Fix  Maxus coordinates
fp = '/home/zachuhlmann/projects/Hedrick_WRR_2018/maxus.nc'
file = h5py.File(fp)

x = file['x']
y = file['y']
dx = x[1] - x[0]
dy = y[1] - y[0]
nlines = len(x)
nsamp = len(y)
# Coordinates are marked at BinCenter.  Y coordinates were also generated incorrectly
# Created from min Y and generated like range(y_start, nsamps, -50) instead of (''. ''. 50)
xt = []
[xt.append(np.float(x + 25)) for x in x]
yt = np.int(y[0] + 25)
yt = list(range(yt, yt + (nsamp) * 50, 50))
y = []
[y.append(np.float(yt)) for yt in yt]

vt1 = file['maxus']
print(vt1.shape)
vt2 = np.flip(np.array(vt1), 1)

# fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (6,8), dpi = 180)
# axs[0].imshow(vt1[0,:,:])
# axs[1].imshow(vt2[0,:,:])
# plt.show()

dset = netCDF4.Dataset(fp, 'r+')
dset.variables['x'][:] = xt
dset.variables['y'][:] = y
dset.variables['maxus'][:] = vt2
dset.close
