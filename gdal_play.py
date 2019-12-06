# from spatialnc.topo import get_topo_stats

# fp = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/USCASJ20190614_20190704_diff_normalized_flags.tif'
fp = '/home/zachuhlmann/projects/data/SanJoaquin/USCASJ20190704_SUPERsnow_depth_50p0m_agg_merged.tif'
fp_out = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/test.nc'



# import numpy as np
import rasterio as rio

import numpy as np

fp_50m = '/home/zachuhlmann/projects/data/SanJoaquin/20180422_20180601/USCASJ20180422_20180601_coords_overhaul_flags_temp.tif'
fp_out = '/home/zachuhlmann/projects/data/SanJoaquin/20180422_20180601/test.tif'

# Note flag is dtype float32 since it's a percentage
with rio.open(fp_50m) as src:
    rio_obj = src
    meta = rio_obj.profile
    arr = rio_obj.read()[:]
    band_str = list(rio_obj.descriptions))
pct_list = [0.1, 0.2, 0.3]

for pct in pct_list:
    temp_arr = np.sum((arr > pct) * 1, axis = 0)
    temp_arr = temp_arr[np.newaxis]
    if 'temp_stack' not in locals():
        temp_stack = np.ndarray.astype(temp_arr, 'uint8')
    else:
        temp_stack = np.concatenate \
            ((temp_stack, np.ndarray.astype(temp_arr, 'uint8')), axis = 0)

meta.update({'count':len(pct_list),
            'dtype':'uint8',
            'nodata':255})


pct_list = ['{}% thresh'.formt(int(pct * 100)) for pct in pct_list]
band_str.extent(pct_list)

with rio.open(fp_out, 'w', **meta) as dst:
    for i in len(shape.temp_stack[0]):
        dst.set_band_description(i + 1, '{}% thresh'.format(band_str[i]))
    dst.write(temp_stack)
