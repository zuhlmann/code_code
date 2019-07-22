import rasterio as rio
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import time


class GDAL_python_synergy():
    def __init__(self, fp_d1, fp_d2, fp_out):
        with rio.open(fp_d1) as src:
            self.d1 = src
            self.meta1 = self.d1.profile
            mat1 = self.d1.read()  #matrix
            self.mat1 = mat1[0]
        with rio.open(fp_d2) as src:
            self.d2 = src
            self.meta2 = self.d2.profile
            mat2 = self.d2.read()  #matrix
            self.mat2 = mat2[0]
        self.fp_out = fp_out
    def clip_extent_overlap(self):
        ''' finds overlapping extent of two tiffs'''
        meta = self.meta1  #metadata
        d1 = self.d1
        d2 = self.d2
        #grab basics
        bounds = [d1.bounds.left, d1.bounds.bottom, d1.bounds.right, d1.bounds.top, \
                d2.bounds.left, d2.bounds.bottom, d2.bounds.right, d2.bounds.top]
        rez = meta['transform'][0]  # spatial resolution
        # minimum overlapping extent
        left_max_bound = max(bounds[0], bounds[4])
        bottom_max_bound = max(bounds[1], bounds[5])
        right_min_bound = min(bounds[2], bounds[6])
        top_min_bound = min(bounds[3], bounds[7])
        self.top_max_bound = top_min_bound
        self.left_max_bound = left_max_bound
        for d in range(2):  # datasets
            # below loop makes list of line and sample georaphic coordinates (column and row)
            for r in range(2):  #row or column (i.e. samples or lines)
                num_lines_or_samps = math.ceil((bounds[3 + d * 4 - r] - bounds[1 + d * 4 - r])/rez)
                coords = []
                for c in range(0, num_lines_or_samps):
                    coords.append(bounds[1 + d * 4 - r] + rez * c)
                if r == 0:  # row and data 1
                    del coords[0]
                    coords.append(bounds[1 + d * 4 - r] + rez * (c + 1))
                    row = coords.copy()
                    num_samps = num_lines_or_samps
                elif r == 1:  # col and data 1:
                    col = coords.copy()
            # get indices of minimum overlapping extent
            ids = [None] * 4
            ids[0:4] = [num_samps - row.index(top_min_bound) - 1, num_samps - row.index(bottom_max_bound + rez) -1, \
                    col.index(left_max_bound), col.index(right_min_bound - rez)]
            #subset both datasets --> mat_clip[num]
            if d == 0:
                mat = self.mat1
                mat_clip1 = mat[ids[0]:ids[1], ids[2]:ids[3]]
                # print('dataset 1. bottom: ', ids[0], 'top: ', ids[1], 'left: ', ids[2], 'right: ', ids[3])
                # print('height: ', self.meta1['height'], 'width: ', self.meta1['width'])
            elif d == 1:
                mat = self.mat2
                mat_clip2 = mat[ids[0]:ids[1], ids[2]:ids[3]]
                # print('dataset 2. bottom: ', ids[0], 'top: ', ids[1], 'left: ', ids[2], 'right: ', ids[3])
                # print('height: ', self.meta2['height'], 'width: ', self.meta2['width'])

        # calculations on np arrays fail with np.nan values.  save one of each mat
        # with nans and -9999
        self.mat_clip1_nans, self.mat_clip2_nans = mat_clip1.copy(), mat_clip2.copy()
        mat_clip1, mat_clip2 = mat_clip1.copy(), mat_clip2.copy()
        mat_clip1[np.isnan(mat_clip1)] = -9999
        mat_clip2[np.isnan(mat_clip2)] = -9999
        self.mat_clip1, self.mat_clip2 = mat_clip1, mat_clip2

    def make_diff_mat(self):
        mat_clip1 = self.mat_clip1.copy()
        mat_clip2 = self.mat_clip2.copy()

        self.mat_diff = mat_clip2 - mat_clip1
        mat_clip1[(mat_clip1 < 0.25) & (mat_clip1 != -9999)] = 0.25  # To avoid dividing by zero
        mat_diff_norm = np.round((self.mat_diff / mat_clip1), 2)
        self.mat_diff_norm = mat_diff_norm.copy()
        mat_diff_norm_nans = mat_diff_norm.copy()
        mat_diff_norm_nans[(mat_clip1 == -9999) | (mat_clip2 == -9999)] = np.nan
        self.mat_diff_norm_nans = mat_diff_norm_nans.copy()
        # self.mat_clip2 = self.nan_reverse('mat_clip2', 2)

    def mask(self):
        '''consider consolidating with mask_advanced, essentially creates self.overlap'''
        # self.id_present = (self.mat_clip1!=-9999) & (self.mat_clip2!=-9999)
        self.id_nans_mat_clip1 = np.isnan(self.mat_clip1)
        self.id_nans_mat_clip2 = np.isnan(self.mat_clip2)
        self.id_nans_mat_overlap = (np.isnan(self.mat_clip1)) | (np.isnan(self.mat_clip2))

    def nan_reverse(self, str, option):
        mat = getattr(self, str)
        if option ==1:
            mat[isnan(mat)] = -9999
        elif option ==2:
            mat[mat == -9999] = np.nan
        return mat

    def mask_advanced(self, name, op, val):
        ''' returns a boolean where no nans and all comparison conditions are met'''
        # Arguments
        # name:  matrix name as saved in self.[NAME]  Type = string
        # op:  operation codes to be performed on each matrix.  Type = list or tuple
        # val: Sets values to perform comparison operations.  Type = list or tuple (match dimensions of op)
        # OUTPUTS
        # self.overlap_nan    boolean where all matrices do not have nans or -9999
        # self.overlap_conditional    boolean with no nans that meets conditions for all overlapping matrices
        keys = {'lt':'<', 'gt':'>'}
        shp = getattr(self, name[0]).shape
        overlap_nan = np.ones(shp, dtype = bool)
        overlap_conditional = np.ones(shp, dtype = bool)
        for i in range(len(name)):
            mat = getattr(self, name[i])
            try:
                num_op = len(op[i])
            except IndexError:
                num_op = 0
            # replace nan with -9999 and identify location for output
            if np.isnan(mat).any():  # if nans are present and represented by np.nan
                mat_mask = mat.copy()
                temp_nan = ~np.isnan(mat_mask)
                mat_mask[np.isnan[mat_mask]] = -9999  # set nans to -9999 (if they exist)
            elif (mat == -9999).any():  # if nans are present and represented by -9999
                mat_mask = mat.copy()
                temp_nan = mat_mask != -9999
                print('num nans', np.sum(mat_mask==-9999))
            else:   # no nans present
                mat_mask = mat.copy()
                temp_nan = np.ones(shp, dtype = bool)
            for j in range(num_op):
                op_str = keys[op[i][j]]
                cmd = 'mat_mask' + op_str + str(val[i][j])
                temp = eval(cmd)
                # overlap = overlap & temp
                overlap_conditional = overlap_conditional & temp
            # overlap = overlap & temp_nan  # where conditions of comparison are met and no nans
            overlap_nan = overlap_nan & temp_nan  # where conditions of comparison are met and no nans
        self.overlap_nan = overlap_nan.copy()
        self.overlap_conditional = overlap_conditional & overlap_nan
        temp = self.mat_diff_norm[self.overlap_nan]
        self.lb = round(np.nanpercentile(temp[temp < 0], 50), 3)
        self.ub = round(np.nanpercentile(temp[temp > 0], 50), 3)

    def mov_wind(self, name, size):
        ''' moving window base function.  switch filter/kernel for different flag'''
        #INPUT
        # name   string for self.<file_name>
        # size   moving window size, i.e. size = size^2
        # thr    fraction of neighboring pixels present below which, pixel is flagged as outlier
        # Note: currently used for 2dhist, but can be used on any 2d image
        if isinstance(name, str):
            mat = getattr(self, name)
        else:
            mat = name
        nrow, ncol = mat.shape
        base_offset = math.ceil(size/2)
        # flag_spatial = np.zeros(mat.shape, dtype = bool)
        pct = np.zeros(mat.shape, dtype = float)  # proportion of neighboring pixels in mov_wind
        ct = 0
        time_row = []
        for i in range(nrow):
            start = time.time()
            if i >= base_offset - 1:
                prox_row_edge = nrow - i - 1
                if prox_row_edge >= base_offset - 1:
                    row_idx = np.arange(base_offset * (-1) + 1, base_offset)
                elif prox_row_edge < base_offset - 1:
                    prox_row_edge = nrow - i - 1
                    row_idx = np.arange(prox_row_edge * (-1), base_offset) * (-1)
            elif i < base_offset - 1:
                prox_row_edge = i
                row_idx = np.arange(prox_row_edge * (-1), base_offset)
            end = time.time()
            time_row.append(end-start)
            for j in range(ncol):
                if j >= base_offset - 1:
                    prox_col_edge = ncol - j - 1
                    if prox_col_edge >= base_offset - 1:  #no window size adjustment
                        col_idx = np.arange(base_offset * (-1) + 1, base_offset)
                    elif prox_col_edge < base_offset - 1:  #at far column edge. adjust window size
                        prox_col_edge = ncol - j - 1
                        col_idx = np.arange(prox_col_edge * (-1), base_offset) * (-1)
                if j < base_offset - 1:
                    prox_col_edge = j
                    col_idx = np.arange(prox_col_edge * (-1), base_offset)

                # Begin the real stuff
                base_row = np.ravel(np.tile(row_idx, (len(col_idx),1)), order = 'F') + i
                base_col = np.ravel(np.tile(col_idx, (len(row_idx),1))) + j
                sub = mat[base_row, base_col]
                pct[i,j] = (np.sum(sub > 0)) / sub.shape[0]
        # self.pct = pct    #deleted
        print(self.mean(time_row))
        return(pct)


    def hist_utils(self, name, nbins):
        '''consider adding density option to np.histogram2d'''
        # InPUTS
        # name   list of strings ['<name1>', '<name2>'] which are x and y respectively in 2dhist
        # nbins  (nybins, nxbins)
        # I don't think an array with nested tuples is computationally efficient.  Find better data structure for the tuple_array
        m1, m2 = getattr(self, name[0]), getattr(self, name[1])
        self.mat_shape = m1.shape
        m1_nan, m2_nan = m1[self.overlap_conditional], m2[self.overlap_conditional]
        bins, xedges, yedges = np.histogram2d(np.ravel(m1_nan), np.ravel(m2_nan), nbins)
        # Now find bin edges of overlapping snow depth locations from both dates, and save to self.bin_loc as array of tuples
        xedges = np.delete(xedges, -1)   # remove the last edge
        yedges = np.delete(yedges, -1)
        bins = np.flip(np.rot90(bins,1), 0)  # WTF np.histogram2d.  hack to fix bin mat orientation
        self.bins, self.xedges, self.yedges = bins, xedges, yedges
        print('num bins: ', bins.shape, 'numedges: ', yedges.shape, xedges.shape)
        # Note: subtract 1 to go from 1 to N to 0 to N - 1 (makes indexing possible below)
        idxb = np.digitize(m1_nan, xedges) -1  # id of x bin edges.  dims = (N,) array
        idyb = np.digitize(m2_nan, yedges) -1  # id of y bin edges
        tuple_array = np.empty((nbins[1], nbins[0]), dtype = object)
        id = np.where(self.overlap_conditional)  # coordinate locations of mat used to generate hist
        idmr, idmc = id[0], id[1]  # idmat row and col
        for i in range(idyb.shape[0]):
                if type(tuple_array[idyb[i], idxb[i]]) != list:  #initiate list if does not exist
                    tuple_array[idyb[i], idxb[i]] = []
                tuple_array[idyb[i], idxb[i]].append([idmr[i], idmc[i]])  #appends map space indices into bin space
        self.bin_loc = tuple_array  #array of tuples containing 0 to N x,y coordinates of overlapping snow map
                                    #locations contributing to 2d histogram bins
        print('finished hist')
    def outliers_hist(self, thresh, moving_window_name, moving_window_size):
        '''creates boolean of all 2d histogram-space outliers (self.outliers_hist_space)'''
        # INPUTS
        # thr   threshold of bin counts below which outliers will be flagged
        flag_bin_ct = self.bins < thresh[0]
        # Prob change:  flag of melt above threshold i.e. > (thr * 100) %
        # mnr = np.max(np.where(self.yedges <= thr[1]))  #should be row where meltout occured above thresh
        # flag_complete_melt = np.zeros(self.bins.shape, dtype=bool)
        # flag_complete_melt[:mnr+1,:] = True
        # flag_complete_melt = flag_complete_melt & (self.bins > 0)
        pct = self.mov_wind(moving_window_name, moving_window_size)
        flag_spatial_outlier = (pct < thresh[2]) & (self.bins>0)
        flag = (flag_spatial_outlier | flag_bin_ct)
        self.outliers_hist_space = flag
        self.hist_to_map_space()  # unpack hisogram space outliers to geographic space locations

    def hist_to_map_space(self):
        ''' unpacks histogram bins onto their contributing map locations (self.outliers_hist_space)'''
        hist_outliers = np.zeros(self.mat_shape, dtype = int)
        # idarr = np.where(pd.notna(self.bin_loc))  # id where bin array has a tuple of locations
        idarr = np.where(self.outliers_hist_space)
        for i in range(len(idarr[0])):  # iterate through all bins with tuples
            loc_tuple = self.bin_loc[idarr[0][i], idarr[1][i]]
            for j in range(len(loc_tuple)):  # iterate through each tuple within each bin
                pair = loc_tuple[j]
                hist_outliers[pair[0], pair[1]] = 1
        self.outliers_map_space = hist_outliers  # unpacked map-space locations of outliers

    def outliers_map(self):
        if hasattr(self, 'outliers_map_space'):
            self.outliers_map_space = (self.flag_block | self.outliers_map_space)
        else:
            self.outliers_map_space = copy.deepcopy(self.flag_block)
    def block_behavior(self, moving_window_size, neighborhood_threshold):
        all_loss = 1 * (self.mat_clip1 != 0) & (self.mat_clip2 == 0)  #gained everything
        all_gain = 1 * (self.mat_clip1 == 0) & (self.mat_clip2 != 0)  #lost everything
        loss_outliers = self.mat_diff_norm < self.lb
        gain_outliers = self.mat_diff_norm > self.ub
        flag_loss_block = all_loss & loss_outliers
        flag_gain_block = all_gain & gain_outliers
        print('this may take awhile!')
        start = time.time()
        pct = self.mov_wind(flag_loss_block, moving_window_size)
        end = time.time()
        print('first block time: ', end-start)
        flag_loss_block = (pct > neighborhood_threshold) & self.overlap_conditional
        self.flag_loss_block = flag_loss_block.copy()
        start = time.time()
        pct = self.mov_wind(flag_gain_block, moving_window_size)
        end = time.time()
        print('second block time: ',end-start)
        flag_gain_block = (pct >neighborhood_threshold) & self.overlap_conditional
        self.flag_gain_block = flag_gain_block.copy()
        self.flag_block = self.flag_gain_block | self.flag_loss_block
        self.flag_block = self.flag_gain_block

    def __repr__(self):
            return ('once filled out this will describe my object')
    # def pattern_flag(self, sub):
    #     size = self.mov_wind_size
    #     pres = sub > 0
    #     base = list(range(size))
    #     base_tuple = []
    #     for i in range(size):
    #         base_tuple.append([base_tuple[i] + num for num in base])
    #     deep_melt_patterns
    #     deep_melt_partial_patterns = [[0,1],[1,2],]

    def trim_extent_nan(self, name):
        '''used to trim path and rows from array edges with no values.  For display'''
        # INPUT
        # name   name of mat to be trimmed to nan extent
        # OUTPUT
        # returns trimmed mat
        mat_out = getattr(self, name)
        mat = self.overlap_nan
        nrows, ncols = self.overlap_nan.shape[0], self.overlap_nan.shape[1]
        #Now get indices to clip excess NAs
        tmp = []
        for i in range(nrows):
            if any(mat[i,:] == True):
                id = i
                break
        tmp.append(id)
        for i in range(nrows-1, 0, -1):  #-1 because of indexing...
            if any(mat[i,:] == True):
                id = i
                break
        tmp.append(id)
        for i in range(ncols):
            if any(mat[:,i] == True):
                id = i
                break
        tmp.append(id)
        for i in range(ncols-1, 0, -1):  #-1 because of indexing...
            if any(mat[:,i] == True):
                id = i
                break
        tmp.append(id)
        idc = tmp.copy()   #idx to clip [min_row, max_row, min_col, max_col]
        mat_out = mat_out[idc[0]:idc[1],idc[2]:idc[3]]
        if ~hasattr(self, 'overlap_nan_trim'):
            self.overlap_nan_trim = self.overlap_nan[idc[0]:idc[1],idc[2]:idc[3]]  # overlap boolean trimmed to nan_extent
        return mat_out
    def row_col_to_idx(self, idp):
        ''' never used.  converts [[<row index array> (N,)], [<column index array> (N,) ]] to array of single value indices'''
        nrow = idp.shape[0]
        ncol = idp.shape[1]
        id = []
        for i in range(nrow):
            for j in range(ncol):
                tmp = ncol * i + j
                id.append(tmp)
        self.id = id
    def save_tiff(self, fname, *argv):
        ''' saves clipped mat to geotiff using RasterIO basically'''
        # INPUT
        # mat   mat to save
        # fname  <filename> string
        # OUTPUT
        # saves a tiff with <filename>.tif to filepath (self.fp_out)
        fn_out = self.fp_out + fname + '.tif'
        if len(argv) > 0:  # if a specific band is specified
            name = argv[0]
            dims = getattr(self, name).shape
            count = 1
        else:
            name = ['flag_loss_block', 'flag_gain_block', 'mat_diff_norm_nans']
            dims = getattr(self, name[0]).shape
            print('dims (hopefully 2d): ', dims)
            count = len(name)

        meta_new = deepcopy(self.meta1)
        aff = deepcopy(self.d1.transform)
        new_aff = rio.Affine(aff.a, aff.b, self.left_max_bound, aff.d, aff.e, self.top_max_bound)
        meta_new.update({
            'height':dims[0],
            'width': dims[1],
            'transform': new_aff,
            'count': count})
        print('meta new: ', meta_new)
        if count == 1:
            with rio.open(fn_out, 'w', **meta_new) as dst:
                mat_temp = getattr(self,name)
                try:
                    dst.write(mat_temp,1)# print(mask.shape)
                except ValueError:
                    dst.write(mat_temp.astype('float32'),1)
        if count > 1:
            with rio.open(fn_out, 'w', **meta_new) as dst:
                for id, band in enumerate(name, start = 1):
                    try:
                        dst.write_band(id, getattr(self, name[id - 1]))# print(mask.shape)
                    except ValueError:
                        mat_temp = getattr(self, name[id - 1])
                        dst.write_band(id, mat_temp.astype('float32'))
    def replace_qml(self):
        ''' modifies qml file from qgis to replace color ramp with one scaled to mat'''
        import math
        num_inc = 3
        linc = round(math.ceil(self.lb * 10) / (10 * (num_inc)), 2)
        uinc = round(math.floor(self.ub * 10) / (10 * (num_inc)), 2)
        lr = list(np.arange(0, self.lb, linc))
        ur = list(np.arange(0, self.ub, uinc))

        col_ramp = lr[1:len(lr)] + [0] + ur[1:len(ur)]
        col_ramp.sort()
        col_ramp = [round(col_ramp, 2) for col_ramp in col_ramp]

        fp = '/home/zachuhlmann/projects/basin_masks/color_ramp_diagnose2.qml'
        fp_out = '/home/zachuhlmann/projects/basin_masks/color_ramp_diagnose.qml'

        lst_old = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
        lst_new = col_ramp
        str_nums = [str(num) for num in lst_old]
        str_nums2 = [str(num) for num in lst_new]

        zip_lst = list(zip(str_nums, str_nums2))

        check = True
        i = 0  # counter
        with open(fp) as infile, open(fp_out, 'w') as outfile:
            for line in infile:
                if check == True:
                    strf = 'label="' + zip_lst[i][0]
                    print('strf ', strf)
                    if  strf in line:
                        print(i)
                        line = line.replace(zip_lst[i][0], zip_lst[i][1])
                        print(line)
                        i += 1
                        if i == len(zip_lst):
                            check = False
                outfile.write(line)
    def buff_points(self, buff):
        ''' used as input to curve fitting.  finds max bin height in y direction and saves coords'''
        # row_mx = max bin heightflag_complete_melt
        # col_max = x locations
        nrow, ncol = self.mat_out.shape
        row_idx = np.arange(nrow)
        row_mx = []
        col_mx = []
        for i in range(ncol):
            try:
                row_mx.append(np.max(row_idx[self.flag2[:,i]]))
                col_mx.append(i)
            except ValueError:
                # row_mx[i] = 0
                pass

        row_mx, col_mx = np.array(row_mx, dtype = int), np.array(col_mx, dtype = int)
        # when buffering, index might exceed xbin and ybin later.  Remove final col_mx and decrease highest row_mx
        if buff > 0:
            row_mx, col_mx = row_mx[:-buff], col_mx[:-buff]
            if any(row_mx >= nrow - buff):
                row_mx[row_mx >= nrow - buff] = nrow - buff - 1
        col_mx_orig, row_mx_orig = col_mx.copy(), row_mx.copy()
        col_mx, row_mx = col_mx + buff, row_mx + buff
        self.col_mx, self.row_mx, self.col_mx_orig, self.row_mx_orig = col_mx, row_mx, col_mx_orig, row_mx_orig

    def thresh_twoD_hist(self, hist_array):
        '''not sure if needed anymore.  used to display clipped 2d hist from plt.hist2d'''
        hist_array = hist_array[(hist_array>5) & (hist_array<2000)]
        hist, bins = np.histogram(hist_array, bins = 30)
        fig,ax = plt.subplots(nrows=1,ncols=1)
        ax.bar(bins[:-1], hist, width = bins[2] - bins[1])
        # plt.ylim(min(hist), 1.1 * max(hist))
        plt.show()

    def fractional_exp(self, x, a, c, d):
        return a * np.exp(-c * x) + d
    def mean(self, numbers):
        return float(sum(numbers)) / max(len(numbers), 1)
