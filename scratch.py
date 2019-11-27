# import numpy as np
# import rasterio as rio
# import matplotlib.pyplot as plt
# from skimage import filters
#
# fp_snownc1 = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/run20190613_snow.nc'
# fp_snownc2 = '/home/zachuhlmann/projects/data/SanJoaquin/20190614_20190704/run20190614_snow.nc'
#
# fp_snownc1_rio = 'netcdf:{}:{}'.format(fp_snownc1, 'thickness')
# with rio.open(fp_snownc1_rio) as src:
#     d1 = src
#     d1 = d1.read()
#     d1 = d1[0]
# fp_snownc2_rio = 'netcdf:{}:{}'.format(fp_snownc2, 'thickness')
# with rio.open(fp_snownc2_rio) as src:
#     d2 = src
#     d2 = d2.read()
#     d2 = d2[0]
#
# diff_arr = d1 - d2
# pct = np.sum(~np.isnan(diff_arr))
# num_outliers = 6000
# pct = (1 - (num_outliers / pct)) *100
# thresh = np.nanpercentile(diff_arr, pct)
# diff_arr[diff_arr < thresh] = thresh
#
# edges = filters.sobel(diff_arr)
#
# low = 0.1
# high = 0.35
#
# lowt = (edges > low).astype(int)
# hight = (edges > high).astype(int)
# hyst = filters.apply_hysteresis_threshold(edges, low, high)
#
# fig, ax = plt.subplots(nrows=2, ncols=2)
#
# ax[0, 0].imshow(diff_arr, cmap='gray')
# ax[0, 0].set_title('Original image')
#
# ax[0, 1].imshow(edges, cmap='magma')
# ax[0, 1].set_title('Sobel edges')
#
# ax[1, 0].imshow(lowt, cmap='magma')
# ax[1, 0].set_title('Low threshold')
#
# ax[1, 1].imshow(hight + hyst, cmap='magma')
# ax[1, 1].set_title('Hysteresis threshold')
#
# for a in ax.ravel():
#     a.axis('off')
#
# plt.tight_layout()
#
# plt.show()
ids = []

for id in list(range(0,100,2)):
    for id2 in list(range(1,100,2)):
        print('id ', id)
        if id2 > 8:
            ids.append(id)
            break
print(ids)
