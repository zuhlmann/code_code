import h5py
from smrf import ipw
import numpy as np
import matplotlib.pyplot as plt
import getCDF as gcdf

fp_topo = '/home/zachuhlmann/projects/Hedrick_WRR_2018/tuol_topo_wrr18.nc'
fp_veg_type = '/home/zachuhlmann/projects/zenodo_WRR_data/static_grids/tuolx_vegnlcd_50m.ipw'

gcdf_obj = gcdf.GetCDF()
gcdf_obj.get_topo(fp_topo)
ipw_array = ipw.IPW(fp_veg_type).bands[0].data
ipw_array = ipw_array.astype(np.int8)
nc_array = np.array(gcdf_obj.topo['veg_type'])


# ipw_array[gcdf_obj.mask == False] = np.nan
ipw_array = np.ma.masked_where(gcdf_obj.mask == False, nc_array)
ipw_array = ipw_array[gcdf_obj.trim_to_NA_extent()]
ipw_array = np.reshape(ipw_array, (gcdf_obj.nrows_trim, gcdf_obj.ncols_trim))
nc_array = np.ma.masked_where(gcdf_obj.mask == False, nc_array)
nc_array = nc_array[gcdf_obj.trim_to_NA_extent()]
nc_array = np.reshape(nc_array, (gcdf_obj.nrows_trim, gcdf_obj.ncols_trim))

fig, axs = plt.subplots(ncols =3, nrows =1)
axs[0].imshow(ipw_array)
axs[1].imshow(nc_array)
mp = axs[2].imshow(ipw_array - nc_array)
fig.colorbar(mp, ax=axs[2], fraction=0.04, pad=0.04, orientation = 'horizontal', extend = 'max')

plt.show()
