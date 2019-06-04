import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotables as pltz
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from snowav.utils.MidpointNormalize import MidpointNormalize

class GetCDF():
    """fill this in"""
    def __init__(self, netCDF, string, obs_start, event_start, nc_time_delt):
    # netCDF = the filepath for netCDF of interest.
    # String = the name of variable that pulls the rasters
    # nc_time_delt  = 'h' for inputs (i.e. precip) and 'd' for ouputs(i.e. specific mass)
    # ids = id of event start in 'days from time_st'
    # dt = datetime list for entire observation period
        self.netCDF = h5py.File(netCDF, 'r')
        self.string = string
        self.nrows = self.netCDF[string].shape[1]
        self.ncols = self.netCDF[string].shape[2]
        self.nobs = self.netCDF[string].shape[0]
        self.nc_time_delt = nc_time_delt
        time_st = pd.to_datetime(obs_start)
        event_st = pd.to_datetime(event_start)
        time_unit = list(range(0, self.nobs, 1))
        self.dt = [time_st + pd.to_timedelta(t, unit = nc_time_delt) for t in time_unit]
        ids = [i for i,x in enumerate(self.dt) if x == event_st]
        self.ids = int(ids[0])

    def mask(self, mask):
    # mask = mask.nc for basin
        mask = h5py.File(mask, 'r')
        mask = mask['mask']
        mask = np.array(mask, dtype = bool)
        self.mask = mask
        #Now get indices to clip excess NAs
        mat = self.mask
        tmp = []
        for i in range(self.nrows):
            if any(mat[i,:] == True):
                idt = i
                break
        tmp.append(idt)
        for i in range(self.nrows-1, 0, -1):  #-1 because of indexing...
            if any(mat[i,:] == True):
                idt = i
                break
        tmp.append(idt)
        for i in range(self.ncols):
            if any(mat[:,i] == True):
                idt = i
                break
        tmp.append(idt)
        for i in range(self.ncols-1, 0, -1):  #-1 because of indexing...
            if any(mat[:,i] == True):
                idt = i
                break
        tmp.append(idt)
        self.idc = tmp   #idx to clip [min_row, max_row, min_col, max_col]
    def print_dates(self, num_times, incr):
        ''' Note: this method necessary to get self.idt for use in rest of class '''
    # num_times = number of time slices to take beginning at ids in increments of incr
    # idt = idx of selected times from num_times
    # incr = increments to select times (days for snow.nc, hours for smrf nc)
        idt = [None] * num_times
        idt = [self.ids + incr * i for i in range(num_times)]  # date indices
        # print('The dates are:')
        # for j in idt:
        #     print(self.dt[j])
        self.idt = idt
        self.num_times = num_times

    def pull_times(self, pds, di, hd):
        ''' Note:  this has not been updated since overhaul, remodel and change of __init__ '''
    #dt = time objects for each mat (from find_ids) # pds = periods, di = day interval, hd = hours daily
    # Outputs: dt_pds = time start for each subplot. m_out = vector(?) of all images for entire period
    # if using snow.nc or other daily outputs (not hourly), set hd = 1
    # adding pds,di and hd to object for use later
        self.pds = pds
        self.di = di
        self.hd = hd
        pix = self.ncols * self.nrows
        # if else ensures proper indexing.  'h' for smrf inputs, 'd' for smrf outputs
        if self.nc_time_delt == 'h':
            hrs = 24
        elif self.nc_time_delt == 'd':
            hrs = 1
        sz = pix * pds * di * hd
        m_out = np.zeros(sz)
        mat = np.zeros((pds*di*hd, self.nrows, self.ncols))
        idi = 0  #index initial for m_out
        ct = 0   #index initial for mat
        dt_pds = [0]*pds
        for t in np.arange(pds):
            dt_pds[t] = self.dt[self.ids + (t*di)*hrs]
            for u in np.arange(di):
                mat2 = self.netCDF[self.string][self.ids + t*di*hrs + u*hrs: self.ids + t*di*hrs +u*hrs + hd]
                mat[ct:ct+hd,:,:] = mat2[0:hd,:,:]
                mat2 = mat2.flatten()
                ide = idi + hd * pix
                m_out[idi:ide] = mat2
                idi = ide #keeps counter going ide = id end; idi = id initial
                ct += hd
        self.m_out = m_out
        self.mat = mat
        self.dt_pds = dt_pds

    def plot_CDF(self, img_str):
        ''' to be used with pull_times method. Not yet generalized '''
        pltz_obj = pltz.Plotables()
        pltz_obj.dist_subP(self.pds)
        ct, ct_in = 0, 0
        for i in range(len(pltz_obj.row)):
            fig, axs = plt.subplots(nrows = pltz_obj.row[i], ncols = pltz_obj.col[i], figsize = (6,8), dpi = 180)
            axs = axs.ravel()
            for j,axo in enumerate(axs):
                print(ct_in)
                if not (pltz_obj.panel_flag + (ct_in == self.pds) == 2): #funky way of saying BOTH need to be true
                    temp_daily = self.mat[ct:ct+self.hd*self.di,:,:]
                    time_avg = np.apply_along_axis(np.mean, 0, temp_daily)
                    pltz_obj.cb_readable(time_avg, 4)
                    ct += self.hd*self.di  #counter used to index
                    mp = axo.imshow(time_avg)
                    # cbar = fig.colorbar(mp, ax=axo, fraction=0.04, pad=0.04, orientation = 'horizontal',
                    #                     extend = 'max', format = '%.1f', ticks = pltz_obj.cb_range)
                    cbar = fig.colorbar(mp, ax=axo, fraction=0.04, pad=0.04, orientation = 'horizontal',
                                        extend = 'max', ticks = pltz_obj.cb_range)
                    mp.axes.get_xaxis().set_ticks([])
                    mp.axes.get_yaxis().set_ticks([])
                    mp.axes.set_title(self.dt_pds[ct_in])
                    ct_in += 1
                else:
                    pass
            str = img_str + '%r.png' %(i+1)
            plt.tight_layout()
            plt.savefig(str, dpi=180)

    def plot_diff(self, gcdf_obj_comp):
        '''this one plots 3 col x [n] num_times: old mod, new mod, plus delta'''
    # gcdf_obj_comp = the object of comparison. obtained using getCDF i.e. __init__
        xlabel_hack = ['Modern Model', 'Hedrick WRR18', 'Diff (modern - WRR18)']
        num_times = self.num_times
        mat = np.empty([3 * num_times, self.nrows, self.ncols])
        im = [None] * num_times
        for i in range(num_times):
            im[i] = i * 3   #mat index
            mat_t = self.netCDF[self.string][self.idt[i]]
            mat_t[self.mask == False] = np.nan
            mat[im[i],:,:] = mat_t
            mat_t = gcdf_obj_comp.netCDF[gcdf_obj_comp.string][self.idt[i]]
            mat_t[self.mask == False] = np.nan
            mat[im[i] + 1,:,:] = mat_t
            mat[im[i] + 2,:,:] = mat[i,:,:] - mat[i +1,:,:]  # difference mat
        pltz_obj = pltz.Plotables()
        # fig, axs = plt.subplots(nrows = num_times, ncols = 3, figsize = (8, math.ceil(10 * (num_times/3))), dpi = 180)
        fig, axs = plt.subplots(nrows = num_times, ncols = 3, figsize = (12, 12), dpi = 180)
        ct = 0
        for i, row in enumerate(axs):
            for j, cell in enumerate(row):
                print(ct)
                mp = cell.imshow(mat[ct,:,:])
                pltz_obj.cb_readable(mat[ct,:,:], 4)
                cbar = fig.colorbar(mp, ax=cell, fraction=0.04, pad=0.04, orientation = 'vertical', extend = 'max', ticks = pltz_obj.cb_range)
                ct += 1
                # print(i,j)
                if i == len(axs) - 1:
                    # print(i)
                    cell.set_xlabel(xlabel_hack[j])
                if j == 0:
                    cell.set_ylabel(self.dt[self.idt[i]])
                cell.set_xticklabels([])
                cell.set_yticklabels([])
        # plt.subplot_tool()
        plt.savefig('Hedrick_Comparison_bets.png', dpi=180)
        # plt.show()

    def get_diff(self, gcdf_obj_comp):
        ''' Creates self.diff_mat for use in plot_diff_simple '''
    # diff_mat = 3 x num_times, nrows, ncols array with new, old, delta rep
    # TAF_delt = 1 x num_times list of total basin delta in acre_feet
        num_times = self.num_times
        mat = np.empty([3 * num_times, self.nrows, self.ncols])
        im = [None] * num_times
        acre_ft_diff = []
        acre_ft_diff_norm = []
        avg_spec_mass_orig = []
        # conv_acre_ft = 0.0328084 *
        for i in range(num_times):
            im[i] = i * 3   #mat index
            mat_t = self.netCDF[self.string][self.idt[i]]
            mat_t[self.mask == False] = np.nan   # set masked pixels to nan
            mat[im[i],:,:] = mat_t
            mat_t = gcdf_obj_comp.netCDF[gcdf_obj_comp.string][self.idt[i]]
            mat_t[self.mask == False] = np.nan
            mat[im[i] + 1,:,:] = mat_t
            mat_t = mat[im[i],:,:] - mat[im[i] +1,:,:]  # difference mat
            mat[im[i] + 2,:,:] = mat_t
            spec_mass_all_cells = np.nansum(mat_t)
            spec_mass_all_cells_new = np.nansum(mat[im[i],:,:])
            spec_mass_all_cells_new_avg = np.nanmean(mat[im[i],:,:])
            acre_ft_diff.append(round(self.basin_conversions(spec_mass_all_cells),2))
            acre_ft_diff_norm.append(round((spec_mass_all_cells / spec_mass_all_cells_new) * 100, 2))
            avg_spec_mass_orig.append(round(spec_mass_all_cells_new_avg, 2))
        self.diff_mat_no_trim = mat
        mat = mat[self.trim_to_NA_extent()]
        # hack to reshape trimmed mat.  above step creates a 1d vector of array values
        mat = np.reshape(mat, (self.num_times * 3, self.nrows_trim, self.ncols_trim))
        self.diff_mat = mat
        self.acre_feet_delt = acre_ft_diff
        self.acre_feet_delt_norm = acre_ft_diff_norm
        self.avg_spec_mass_orig = avg_spec_mass_orig

    def plot_diff_simple(self, idi, img_str):
        '''Plots only the delta map in multipanels'''
    # idi options:  2 = diff, 1 = old, 0 = new  InDexIndex (idi)
        num_ticks = 3 # tick marks in colorbar
        fsize = 10  # base font size
        pltz_obj = pltz.Plotables()
        pltz_obj.dist_subP(self.num_times)
        pltz_obj.marks_colors()
        idd = list(range(idi, idi + self.num_times *3, 3))  #idi picks what to plot (new, old, delta)
        if idi == 2:
            col_lim_type = 'A'  #
        else:
            # Find min, max mat vals for all times
            vmn, vmx = 0, 0  # initiate vmx, vmn. WILL BREAK if min is pos or max negative
            for idd in idd:
                mat_d = self.diff_mat[idd,:,:]
                vmn = min(vmn, np.nanmin(mat_d))
                vmx = max(vmx, np.nanmax(mat_d))
            col_lim=[vmn, vmx]
            col_lim_type = 'L'
            pltz_obj.cb_readable(col_lim, col_lim_type, num_ticks)
        ct = 0
        for i in range(len(pltz_obj.row)):
            fig, axs = plt.subplots(nrows = pltz_obj.row[i], ncols = pltz_obj.col[i], figsize = (12,12), dpi = 180)
            axs = axs.ravel()
            for j,axo in enumerate(axs):
                print(ct*3 + idi)
                diff_mat = self.diff_mat[ct*3 + idi,:,:]
                if col_lim_type == 'A':
                    pltz_obj.cb_readable(diff_mat, col_lim_type, num_ticks)
                    mp = axo.imshow(diff_mat, cmap = pltz_obj.cmap, norm = MidpointNormalize(midpoint = 0, vmin = pltz_obj.vmin, vmax = pltz_obj.vmax))
                    cbar = fig.colorbar(mp, ax=axo, fraction=0.04, pad=0.04,
                                        extend = 'max', ticks = pltz_obj.cb_range)
                    cbar.ax.tick_params(labelsize = fsize)
                    axo.set_title(self.dt[self.idt[j]], fontsize=fsize+4)
                elif col_lim_type == 'L':
                    mp = axo.imshow(diff_mat, cmap = pltz_obj.cmap, norm = MidpointNormalize(midpoint = 0, vmin = pltz_obj.vmin, vmax = pltz_obj.vmax))
                    cbar = fig.colorbar(mp, ax=axo, fraction=0.04, pad=0.04, orientation = 'horizontal',
                                        extend = 'max', ticks = pltz_obj.cb_range)
                    cbar.ax.tick_params(labelsize = fsize)
                mp.axes.get_xaxis().set_ticks([])
                mp.axes.get_yaxis().set_ticks([])
                fig.suptitle('Specific Mass (mm) difference map (New Model - Old Model)', fontsize = fsize+8)
                ct += 1
            str = img_str + '%r.png' %(i+1)  # adds +1 to fig name if multi image (i.e. > 9 subplots)
            plt.savefig(str, dpi=180)
            # plt.show()

    def plot_simple(self):
        ''' Note sure if ever used.  may need fix or delete '''
        num_times = len(self.idt)
        fig, axs = plt.subplots(nrows = math.ceil(num_times/2), ncols = 2, figsize = (6,8), dpi = 180)
        axs = axs.ravel()
        for j, axo in enumerate(axs):
            if num_times - 1 >= j:
                axo.imshow(self.netCDF[self.string][self.idt[j]])
                axo.set_axis_off()
        plt.show()

    def basin_conversions(self, spec_mass):
        ''' double check this work '''
    # spec_mass = total specific mass for basin (mm)
        m_ft = 3.0328084  # meters to feet conv
        mm_m = 0.001   # cm to meters conv
        m2_acre = 0.000247105   # m2 to acres conv
        ncells = np.sum(self.mask==True)   # total cells in basin
        cell_size = 50**2   # 50 cell sizes
        spec_mass_ft_avg = (spec_mass * mm_m * m_ft) / ncells
        basin_area = ncells * cell_size * m2_acre
        acre_ft = spec_mass_ft_avg * basin_area
        self.basin_area = basin_area
        return acre_ft
    def trim_to_NA_extent(self):
        # idc indicates extents of basin from mask
        mnr, mxr, mnc, mxc = self.idc[0], self.idc[1], self.idc[2], self.idc[3]
        # make a boolean of False matching array size
        trim_mat = np.full((len(self.idt) * 3, self.nrows, self.ncols), False, dtype = bool)
        trim_mat[:, mnr:mxr, mnc:mxc] = True  #Add True inside trim extent
        # these will be the dimensions once trim mask is employed
        self.nrows_trim = mxr - mnr
        self.ncols_trim = mxc - mnc
        return trim_mat
