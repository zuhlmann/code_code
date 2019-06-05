

class GetCDF_topo():

    def __init__(self, fp):
        self.netCDF = h5py.File(fp, 'r')
        self.nrows = self.netCDF.shape[0]
        self.ncols = self.netCDF.shape[1]

    def mask(self):
    # mask = mask.nc for basin
        mask = self.netCDF['mask']
        self.mask = np.array(mask, dtype = bool)
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

    def trim_to_NA_extent(self):
        trim_mat = np.full((self.nrows, self.ncols), False, dtype = bool)
        mnr, mxr, mnc, mxc = self.idc[0], self.idc[1], self.idc[2], self.idc[3]
        trim_mat[mnr:mxr, mnc:mxc] = True  #Add True inside trim extent
        # these will be the dimensions once trim mask is employed
        self.nrows_trim = mxr - mnr
        self.ncols_trim = mxc - mnc
