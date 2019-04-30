import time_munge as tm
import h5py

# st = '2017-04-01'
# end = '2017-06-30'
# # tm_obj = tm.DispTimeSeries(st,end)
# # print(tm_obj.time)
#
# # h5_obj = h5py.File('../../projects/learning/brb_z/data/output/brb/devel/wy2017/awsm_paper/runs/run4368_6528/em.nc', 'r')
# h5_obj = h5py.File('../../projects/learning/brb_z/data/output/brb/devel/wy2017/awsm_paper/data/data4368_6528/smrfOutputs/precip.nc', 'r')
# # data = h5_obj["SWI"]
# data = h5_obj["precip"]
# print(data.shape[0])
for i in range(3):
    img_str = '4_1_17_to_5_4_17_4hr_inc'
    str = img_str + '%r.png' %(i+1)
    print(str)
