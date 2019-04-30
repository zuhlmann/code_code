import numpy as np
import math

#plottables

class Plotables:
    "this does some stuff"
    def cb_readable(self,array,num_ticks):
        ''' this just gives tick marks plotting '''
        self.array=array
        self.num_ticks = num_ticks
        mn = np.amin(self.array)
        mx = np.amax(self.array)
        if mx-mn<0.1:
          self.cb_range = np.array([mn, mx])
        else:
          self.cb_range = np.linspace(mn*1.1, mx*0.9, self.num_ticks)
    def dist_subP(self, num_subP):
        '''sets distribution of subplot rows and columns based on number of
        subplots'''
        self.panel_flag = not ((num_subP % 9) % 2 == 0)  #Let's user know that there is one empty subplot
        # which should not be plotted later on as slicing will throw errors
        num_subP_t = num_subP
        imgs = math.ceil(num_subP / 9)
        rw = [0] * imgs
        cl = [0] * imgs
        for i in range(math.ceil(num_subP / 9)):
            if imgs > 1:
                rw[i] = 3
                cl[i] = 3
                imgs = imgs - 1  #counter
            else:
                num_subP = num_subP_t - 9*i
                if num_subP == 1:
                    rw[i] = 1
                    cl[i] = 1
                elif num_subP < 9:
                    rw[i] = math.ceil(num_subP/2)
                    cl[i] = 2
                elif num_subP == 9:
                    rw[i] = 3
                    cl[i] = 3
            self.row = rw
            self.col = cl

    def plt_getCDF(self, png_str):
        for i in range(len(self.row)):
            fig, axs = plt.subplots(nrows = self.row[i], ncols = pltz_obj.col[i], figsize = (6,8), dpi = 180)
            axs = axs.ravel()
            for j,axo in enumerate(axs):
                print(ct_in)
                if not (pltz_obj.panel_flag + (ct_in == pds) == 2): #funky way of saying BOTH need to be true
                    print(ct)
                    temp_daily = gcdf_obj.mat[ct:ct+hd*di,:,:]
                    time_avg = np.apply_along_axis(np.mean, 0, temp_daily)
                    pltz_obj.cb_readable(time_avg, 5)
                    ct += hd*di  #counter used to index
                    mp = axo.imshow(time_avg)
                    cbar = fig.colorbar(mp, ax=axo, fraction=0.04, pad=0.04, orientation = 'horizontal',
                                        extend = 'max', format = '%.1f', ticks = pltz_obj.cb_range)
                    mp.axes.get_xaxis().set_ticks([])
                    mp.axes.get_yaxis().set_ticks([])
                    mp.axes.set_title(gcdf_obj.dt_pds[ct_in])
                    ct_in += 1
                else:
                    pass
            str = 'brb_2017_3day_panel%r.png' %(i+1)
            plt.tight_layout()
            plt.savefig(str, dpi=180)
