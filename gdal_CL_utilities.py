import rasterio as rio
import numpy as np
import copy


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
                num_lines_or_samps = int((bounds[3 + d * 4 - r] - bounds[1 + d * 4 - r])/rez)
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
                print('dataset 1. bottom: ', ids[0], 'top: ', ids[1], 'left: ', ids[2], 'right: ', ids[3])
                print('height: ', self.meta1['height'], 'width: ', self.meta1['width'])
            elif d == 1:
                mat = self.mat2[0]
                mat_clip2 = mat[ids[0]:ids[1], ids[2]:ids[3]]
                print('dataset 2. bottom: ', ids[0], 'top: ', ids[1], 'left: ', ids[2], 'right: ', ids[3])
                print('height: ', self.meta2['height'], 'width: ', self.meta2['width'])
        self.mat_clip1 = mat_clip1
        self.mat_clip2 = mat_clip2

    def diagnose(self, thr):
        mat_clip1 = self.mat_clip1.copy()
        mat_clip2 = self.mat_clip2.copy()
        dp_min = 0.5

        self.mat_diff = mat_clip1 - mat_clip2

        print('number of elements = 0: ', np.sum(mat_clip1==0))
        print('mat diff: ', type(self.mat_diff[0,0]), 'mat_clip1: ', type(mat_clip1[0,0]))

        self.mask()
        mat_clip2[self.id_zero_mat2] = 0.25  # To avoid dividing by zero
        mat_diff_norm = np.round((self.mat_diff / mat_clip2), 2)
        self.lb = round(np.nanpercentile(mat_diff_norm[self.id_present], 5  ),3)
        self.ub = round(np.nanpercentile(mat_diff_norm[self.id_present], 99.9),3)
        # mat_diff_norm[mat_diff_norm<0] = mat_diff_norm[mat_diff_norm<0] / (self.lb * -1)
        # mat_diff_norm[mat_diff_norm>=0] = mat_diff_norm[mat_diff_norm>=0] / self.ub
        print('quant :', self.lb, self.ub)
        mat_diff_norm[self.id_nan] = np.nan
        self.mat_diff_norm = mat_diff_norm
        # self.mat_clip2 = self.nan_reverse('mat_clip2', 2)

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
            dst.write(mat,1)

    def nan_ids(self):
        mat = self.mat_clip1
        ids = np.arange(mat[0] * mat[1]).reshape((mat[0], mat[1]))
        idnan = ids[np.isnan(mat)]
        mat_clean = mat[~idnan]
        mat_clean = mat_clean
    def mask(self):
        self.id_bare_gain = np.where((self.mat_diff > 0) & (self.mat_clip1 < 0.5))
        self.id_bare_loss = np.where((self.mat_diff < 0) & (self.mat_clip1 < 0.5))
        self.id_nan = np.where((self.mat_clip1==-9999) | (self.mat_clip2==-9999))
        self.id_present = np.where((self.mat_clip1!=-9999) & (self.mat_clip2!=-9999))
        self.id_zero_mat2 = np.where(self.mat_clip2 < 0.25)
    def nan_reverse(self, str, option):
        mat = getattr(self, str)
        if option ==1:
            mat[isnan(mat)] = -9999
        elif option ==2:
            mat[mat == -9999] = np.nan
        return mat

    def mask_advanced(self, name, op, val):
        keys = {'lt':'<', 'gt':'>', 'fwd':2}
        shp = getattr(self, name[0]).shape
        overlap = np.ones(shp, dtype = bool)
        print('type of shape ', type(overlap))
        for i in range(len(name)):
            print('i ', i)
            mat = getattr(self, name[i])
            num_op = len(op[i])
            print('num_op ', num_op)
            for j in range(num_op):
                op_str = keys[op[i][j]]
                if j < num_op-1:
                    cmd = 'mat' + op_str + str(val[i][j])
                    temp = eval(cmd)
                    print('conditional: ', np.sum(temp)/ (temp.shape[0] * temp.shape[1]))
                elif j == num_op - 1:
                    if op_str == 1:
                        #do reverse
                        pass
                    elif op_str == 2:
                        temp = ~np.isnan(mat)
                        print(type(mat))
                        print('boolnan: ', np.sum(temp) / (temp.shape[0] * temp.shape[1]))
                print('temp type ', type(temp[0]))
                overlap = overlap & temp
                print('overlap: ', np.sum(overlap) / (overlap.shape[0] * overlap.shape[1]))
        return overlap


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
