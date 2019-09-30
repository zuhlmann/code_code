date1 = temp
date2 = temp2

# with rio.open(date1) as src:
#     d1 = src.read()
#     print(d1.shape)
#     d1 = d1[0]
#     meta = src.profile
# with rio.open(date2) as src:
#     d2 = src.read()
#     d2 = d2[0]
#     meta2 = src.profile

 with rio.open(clipped) as src:
     d3 = src.read()
     d3 = d3[0]
     meta3 = d3.profile

if not rasterio.coords.disjoint_bounds(self.d2.bounds, d3.bounds):
    self.disjoint_bounds = False

else:
    self.disjoing_bounds = True
    bounds3 = []
    bounds3[0], bounds3[1] = d3.bounds.left, d3.bounds.right
    bounds3[2], bounds3[3] = d3.bounds.top, d3.bounds.lower

    bounds2 = []
    bounds2[0], bounds2[1] = self.d2.bounds.left, self.d2.bounds.right
    bounds2[2], bounds2[3] = self.d2.bounds.upper, self.d2.bounds.lower

    buffer[0] = round((bounds2[0] - bounds3[0]) / rez) * -1
    buffer[3] = round((bounds2[3] - bounds3[3]) / rez) * -1
    buffer[1] = round((bounds2[1] - bounds3[1]) / rez) * -1
    buffer[2] = round((bounds2[2] - bounds3[2]) / rez) * -1
    self.buffer = buffer

flag_array = np.full(self.d2.shape, 255, dtype = 'uint8')

for i in flags:
    flag_array[top_buffer : bottom_buffer, left_buffer : right_buffer] = flag[i]

#save to tiff of meta2



# grab bounds of common/overlapping extent and prepare function call for gdal to clip to extent and align
left_max_bound = max(d1.bounds.left, d2.bounds.left, topo_extents[0])
bottom_max_bound = max(d1.bounds.bottom, d2.bounds.bottom, topo_extents[1])
right_min_bound =  min(d1.bounds.right, d2.bounds.right, topo_extents[2])
top_min_bound = min(d1.bounds.top, d2.bounds.top, topo_extents[3])
# ensure nothing after decimal - nice whole number, admittedly a float
left_max_bound = left_max_bound - (left_max_bound % round(rez))
bottom_max_bound = bottom_max_bound + (round(rez) - bottom_max_bound % round(rez))
right_min_bound = right_min_bound - (right_min_bound % round(rez))
top_min_bound = top_min_bound + (round(rez) - top_min_bound % round(rez))
