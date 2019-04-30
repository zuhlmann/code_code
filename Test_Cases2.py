import getCDF as gcdf
import plotables as pltz
import matplotlib.pyplot as plt
import numpy as np

# Read in the data, this will take a little bit
# em = h5py.File('../../projects/learning/brb_z/data/output/brb/devel/wy2017/awsm_paper/runs/run4368_6528/em.nc', 'r')
# snow = h5py.File('../../projects/learning/brb_z/data/output/brb/devel/wy2017/awsm_paper/runs/run4368_6528/snow.nc', 'r')
# list(em.keys())  #ZRU if you want to see keys
# get the entire basin mask
# mask_file = h5py.File('../../projects/learning/brb_z/data/input/topo/topo.nc','r')  #ZRU h5py reduces storage space by converting
# mask = mask_file['mask'][:]
# mask = mask_file['mask'][0:4,0:4]  #this should grab rows and cols 1-4, por ejemplo
#To use later
# precip = h5py.File('../../projects/learning/brb_z/data/output/brb/devel/wy2017/awsm_paper/data/data4368_6528/smrfOutputs/precip.nc', 'r')
# temp = h5py.File('../../projects/learning/brb_z/data/output/brb/devel/wy2017/awsm_paper/data/data4368_6528/smrfOutputs/air_temp.nc', 'r')

# to mask by elevation bands
# topo = mask_file["dem"]
# topo = topo[:,:]
# topo[topo<1200]=0
# topo[mask==0]=np.nan

file = '../../projects/learning/brb_z/data/output/brb/devel/wy2017/awsm_paper/data/data4368_6528/smrfOutputs/precip.nc'
gcdf_obj = gcdf.GetCDF(file, "precip", '2017-04-01 00:00:00', '2017-04-01 00:00:00')
print(gcdf_obj.ids)
pds,di,hd = 30, 3, 24
gcdf_obj.pull_times(pds, di, hd)
img_str = '4_1_17_to_6_30_17_24hr'
gcdf_obj.plot_CDF(pds, di, hd, img_str)
