
from raqc import multi_array

raqc_obj = multi_array.MultiArrayOverlap('/home/zachuhlmann/projects/data/USCATE20190613_SUPERsnow_depth_50p0m_agg_clipped.tif',
                                '/home/zachuhlmann/projects/data/USCATE20190705_SUPERsnow_depth_50p0m_agg_clipped.tif',
                                '/home/zachuhlmann/projects/data/')
raqc_obj.clip_extent_overlap()
