import rasterio as rio
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import pandas as pd


class GDAL_python_synergy():
    def __init__(self, fp_d1, fp_d2, fn_out):
        with rio.open(fp_d1) as src:
            self.d1 = src
            self.meta1 = self.d1.profile
            self.mat1 = self.d1.read()  #matrix
        with rio.open(fp_d2) as src:
            self.d2 = src
            self.meta2 = self.d2.profile
            self.mat2 = self.d2.read()  #matrix
        self.fn_out = fn_out
    def clip_min_extent(self):
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
        self.top_max_bound =top_min_bound
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
                mat = self.mat1[0]
                mat_clip1 = mat[ids[0]:ids[1], ids[2]:ids[3]]
                # print('dataset 1. bottom: ', ids[0], 'top: ', ids[1], 'left: ', ids[2], 'right: ', ids[3])
                # print('height: ', self.meta1['height'], 'width: ', self.meta1['width'])
            elif d == 1:
                mat = self.mat2[0]
                mat_clip2 = mat[ids[0]:ids[1], ids[2]:ids[3]]
                # print('dataset 2. bottom: ', ids[0], 'top: ', ids[1], 'left: ', ids[2], 'right: ', ids[3])
                # print('height: ', self.meta2['height'], 'width: ', self.meta2['width'])
        mat_clip1[np.isnan(mat_clip1)] = -9999
        mat_clip2[np.isnan(mat_clip2)] = -9999
        self.mat_clip1 = mat_clip1
        self.mat_clip2 = mat_clip2

    def diagnose(self):
        mat_clip1 = self.mat_clip1.copy()
        mat_clip2 = self.mat_clip2.copy()
        dp_min = 0.5

        self.mat_diff = mat_clip1 - mat_clip2
        self.mask()
        mat_clip2[self.id_zero_mat2] = 0.25  # To avoid dividing by zero
        mat_diff_norm = np.round((self.mat_diff / mat_clip2), 2)
        self.lb = round(np.nanpercentile(mat_diff_norm[self.id_present], 5  ),3)
        self.ub = round(np.nanpercentile(mat_diff_norm[self.id_present], 99.9),3)
        # mat_diff_norm[mat_diff_norm<0] = mat_diff_norm[mat_diff_norm<0] / (self.lb * -1)
        # mat_diff_norm[mat_diff_norm>=0] = mat_diff_norm[mat_diff_norm>=0] / self.ub
        # mat_diff_norm[self.id_nan] = np.nan
        self.mat_diff_norm = mat_diff_norm
        # self.mat_clip2 = self.nan_reverse('mat_clip2', 2)

    def mask(self):
        self.id_present = (self.mat_clip1!=-9999) & (self.mat_clip2!=-9999)
        self.id_zero_mat2 = self.mat_clip2 < 0.25  #NOTE this is where values ARE near ZERO
        temp = (self.mat_clip1!=-9999) & (self.mat_clip2!=-9999)


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
        keys = {'lt':'<', 'gt':'>'}
        shp = getattr(self, name[0]).shape
        overlap = np.ones(shp, dtype = bool)
        for i in range(len(name)):
            mat = getattr(self, name[i])
            num_op = len(op[i])
            # replace nan with -9999 and identify location for output
            if np.isnan(mat).any():  # if nans are present and represented by np.nan
                mat_mask = mat.copy()
                temp_nan = ~np.isnan(mat_mask)
                mat_mask[np.isnan[mat_mask]] = -9999  # set nans to -9999 (if they exist)
            elif (mat == -9999).any():  # if nans are present and represented by -9999
                mat_mask = mat.copy()
                temp_nan = mat_mask != -9999
            else:   # no nans present
                mat_mask = mat.copy()
                temp_nan = np.ones(shp, dtype = bool)
            for j in range(num_op):
                op_str = keys[op[i][j]]
                cmd = 'mat_mask' + op_str + str(val[i][j])
                temp = eval(cmd)
                overlap = overlap & temp
            overlap = overlap & temp_nan  # where conditions of comparison are met and no nans
        self.overlap = overlap

    def mov_wind(self, name, size, thr):
        ''' moving window base function.  switch filter/kernel for different flag'''
        mat = getattr(name)
        nrow, ncol = mat.shape
        base_offset = math.ceil(size/2)
        mat_out = np.zeros(mat.shape)
        mat_out_smooth = mat_out.copy()
        flag = np.zeros(mat.shape, dtype = bool)
        ct = 0
        for i in range(nrow):
            # print('nrow ', i)
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

            for j in range(ncol):
                # print('col ', j)
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
                flag[i,j] = self.spatial_outlier(sub, thr)
                # mat_out[i,j] = mat[i,j]

        mat_out_smooth[flag] = 0
        flag = (mat > 0) & (flag == True)
        self.hist_flag = flag

    def hist_utils(self, name, nbins):
        '''consider adding density option to np.histogram2d'''
        # I don't think an array with nested tuples is computationally efficient.  Find better data structure for the tuple_array
        m1, m2 = getattr(self, name[0]), getattr(self, name[1])
        self.mat_shape = m1.shape
        m1_nan, m2_nan = m1[self.overlap], m2[self.overlap]
        bins, xedges, yedges = np.histogram2d(np.ravel(m1_nan), np.ravel(m2_nan), nbins)
        xedges = np.delete(xedges, -1)   # remove the last edge
        yedges = np.delete(yedges, -1)
        bins = np.flip(np.rot90(bins,1), 0)  # WTF np.histogram2d.  hack to fix bin mat orientation
        self.bins, self.xedges, self.yedges = bins, xedges, yedges
        print('num bins: ', bins.shape, 'numedges: ', xedges.shape, yedges.shape)
        # Note: subtract 1 to go from 1 to N to 0 to N - 1 (makes indexing possible below)
        idxb = np.digitize(m1_nan, xedges) -1  # id of x bin edges.
        idyb = np.digitize(m2_nan, yedges) -1  # id of y bin edges
        tuple_array = np.empty((nbins[1], nbins[0]), dtype = object)
        id = np.where(self.overlap)  # coordinate locations of mat used to generate hist
        idmr, idmc = id[0], id[1]  # idmat row and col
        for i in range(idyb.shape[0]):
                if type(tuple_array[idyb[i], idxb[i]]) != list:
                    tuple_array[idyb[i], idxb[i]] = []
                tuple_array[idyb[i], idxb[i]].append([idmr[i], idmc[i]])
        self.bin_loc = tuple_array
    def thresh_hist(self, thr):
        outliers = (self.bins < thr) & (self.bins != 0)
        self.outliers = outliers

    def map_flagged(self):
        hist_outliers = np.zeros(self.mat_shape, dtype = int)
        # idarr = np.where(pd.notna(self.bin_loc))  # id where bin array has a tuple of locations
        idarr = np.where(self.outliers)
        for i in range(len(idarr[0])):  # iterate through all bins with tuples
            loc_tuple = self.bin_loc[idarr[0][i], idarr[1][i]]
            for j in range(len(loc_tuple)):  # iterate through each tuple within each bin
                pair = loc_tuple[j]
                hist_outliers[pair[0], pair[1]] = 1
        self.hist_outliers = hist_outliers

    def spatial_outlier(self, sub, thr):
        '''flags cells with a relative number of neighboring cells within moving window less than threshold '''
        pct = (np.sum(sub > 0)) / sub.shape[0]
        flag = pct < thr
        return flag
    def buff_points(self, buff):
        ''' used as input to curve fitting.  finds max bin height in y direction and saves coords'''
        # row_mx = max bin height
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
        print(hist)
        print(bins)
        fig,ax = plt.subplots(nrows=1,ncols=1)
        ax.bar(bins[:-1], hist, width = bins[2] - bins[1])
        # plt.ylim(min(hist), 1.1 * max(hist))
        plt.show()

    def basic_stats(self):
        '''probably delete'''
        num_cells = np.sum(self.overlap)
        print('num snow depth cells present: ', num_cells)
        pct999 = round(np.nanpercentile(self.mat_clip1[self.id_present],99),2)
        print(pct999)

    def fractional_exp(self, x, a, c, d):
        return a * np.exp(-c * x) + d

    def save_tiff(self, str):
        meta_new = self.meta1.copy()
        aff = self.d1.transform
        new_aff = rio.Affine(aff.a, aff.b, self.left_max_bound, aff.d, aff.e, self.top_max_bound)
        meta_new.update({
            'height': self.mat_clip1.shape[0],
            'width': self.mat_clip1.shape[1],
            'transform': new_aff})
        # new_aff = rasterio.Affine(aff.a * oview, aff.b, aff.c, aff.c, aff.e * oview, aff.f)
        # meta_new['transform'][0]=20000
        # meta_new['transform'][2], self.meta1['transform'][5] = self.left, self.top
        # print('here')
        mat = getattr(self, str)
        # mat[(mat < self.lb) | (mat > self.up)] = np.nan
        with rio.open(self.fn_out, 'w', **meta_new) as dst:
            try:
                dst.write(mat,1)# print(mask.shape)
            except ValueError:
                dst.write(mat.astype('float32'),1)
    def replace_qml(self):
        import math
        print('upper bound', self.ub)
        num_inc = 3
        linc = round(math.ceil(self.lb * 10) / (10 * (num_inc)), 2)
        uinc = round(math.floor(self.ub * 10) / (10 * (num_inc)), 2)
        print('upper incr ', uinc)
        lr = list(np.arange(0, self.lb, linc))
        ur = list(np.arange(0, self.ub, uinc))
        print('upper range', ur)
        col_ramp = lr[1:len(lr)] + [0] + ur[1:len(ur)]
        col_ramp.sort()
        col_ramp = [round(col_ramp, 2) for col_ramp in col_ramp]
        print('color ramp ', col_ramp)

        fp = '/home/zachuhlmann/projects/basin_masks/color_ramp_diagnose2.qml'
        fp_out = '/home/zachuhlmann/projects/basin_masks/color_ramp_diagnose.qml'

        lst_old = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
        lst_new = col_ramp
        str_nums = [str(num) for num in lst_old]
        str_nums2 = [str(num) for num in lst_new]

        zip_lst = list(zip(str_nums, str_nums2))
        print('zip list ', zip_lst[0][1])

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
