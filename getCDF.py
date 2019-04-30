import h5py
import pandas as pd
import numpy as np
import plotables as pltz
import matplotlib.pyplot as plt

class GetCDF():
    """fill this in"""
    def __init__(self, netCDF, string, obs_start, event_start):
        # netCDF = the filepath for netCDF of interest.  String = the name of variable that pulls the rasters
        self.netCDF = h5py.File(netCDF, 'r')
        self.string = string
        self.nrows = self.netCDF[string].shape[1]
        self.ncols = self.netCDF[string].shape[2]
        self.nobs = self.netCDF[string].shape[0]
        time_st = pd.to_datetime(obs_start)
        event_st = pd.to_datetime(event_start)
        hrs = list(range(0, self.nobs, 1))
        #dt: datetime list for entire observation period
        self.dt = [time_st + pd.to_timedelta(t, unit='h') for t in hrs]
        ids = [i for i,x in enumerate(self.dt) if x == event_st]
        #this is the id start of event
        self.ids = int(ids[0])

    def pull_times(self, pds, di, hd):
        #dt = time objects for each mat (from find_ids) # pds = periods, di = day interval, hd = hours daily
        #Outputs: dt_pds = time start for each subplot. m_out = vector(?) of all images for entire period
        pix = self.ncols * self.nrows
        sz = pix * pds * di * hd
        m_out = np.zeros(sz)
        mat = np.zeros((pds*di*hd, self.nrows, self.ncols))
        idi = 0  #index initial for m_out
        ct = 0   #index initial for mat
        dt_pds = [0]*pds
        for t in np.arange(pds):
            dt_pds[t] = self.dt[self.ids + (t*di)*24]
            for u in np.arange(di):
                mat2 = self.netCDF[self.string][self.ids + t*di*24 + u*24: self.ids + t*di*24 +u*24 + hd]
                mat[ct:ct+hd,:,:] = mat2[0:hd,:,:]
                mat2 = mat2.flatten()
                ide = idi + hd * pix
                m_out[idi:ide] = mat2
                idi = ide #keeps counter going ide = id end; idi = id initial
                ct += hd
        self.m_out = m_out
        self.mat = mat
        self.dt_pds = dt_pds

    def plot_CDF(self, pds, di, hd, img_str):

        pltz_obj = pltz.Plotables()
        pltz_obj.dist_subP(pds)
        ct, ct_in = 0, 0
        for i in range(len(pltz_obj.row)):
            fig, axs = plt.subplots(nrows = pltz_obj.row[i], ncols = pltz_obj.col[i], figsize = (6,8), dpi = 180)
            axs = axs.ravel()
            for j,axo in enumerate(axs):
                print(ct_in)
                if not (pltz_obj.panel_flag + (ct_in == pds) == 2): #funky way of saying BOTH need to be true
                    print(ct)
                    temp_daily = self.mat[ct:ct+hd*di,:,:]
                    time_avg = np.apply_along_axis(np.mean, 0, temp_daily)
                    pltz_obj.cb_readable(time_avg, 5)
                    ct += hd*di  #counter used to index
                    mp = axo.imshow(time_avg)
                    cbar = fig.colorbar(mp, ax=axo, fraction=0.04, pad=0.04, orientation = 'horizontal',
                                        extend = 'max', format = '%.1f', ticks = pltz_obj.cb_range)
                    mp.axes.get_xaxis().set_ticks([])
                    mp.axes.get_yaxis().set_ticks([])
                    mp.axes.set_title(self.dt_pds[ct_in])
                    ct_in += 1
                else:
                    pass
            str = img_str + '%r.png' %(i+1)
            plt.tight_layout()
            plt.savefig(str, dpi=180)
