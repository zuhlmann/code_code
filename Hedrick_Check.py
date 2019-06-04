import h5py
from smrf import ipw
import numpy as np
import matplotlib.pyplot as plt

fp_topo = '/home/zachuhlmann/projects/Hedrick_WRR_2018/tuol_topo_wrr18.nc'
fp_veg_type = '/home/zachuhlmann/projects/zenodo_WRR_data/static_grids/tuolx_vegnlcd_50m.ipw'
topo = h5py.File(fp_topo)
ipw_array = ipw.IPW(fp_veg_type).bands[0].data
nc_array = np.array(topo['veg_type'])

fig, axs = plt.subplots(ncols =3, nrows =1)
axs[0].imshow(ipw_array)
axs[1].imshow(nc_array)
mp = axs[2].imshow(ipw_array - nc_array)
fig.colorbar(mp, ax=axs[2], fraction=0.04, pad=0.04, orientation = 'horizontal', extend = 'max')

plt.show()
