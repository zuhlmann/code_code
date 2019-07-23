pct = self.mov_wind(moving_window_name, moving_window_size)
    flag_spatial_outlier = (pct < thresh[2]) & (self.bins>0)
    flag_bin_ct = self.bins < thresh[0]  #REMOVED
    flag = (flag_spatial_outlier | flag_bin_ct)
    self.outliers_hist_space = flag
    self.hist_to_map_space()  # unpack hisogram spacconsider adding density option to np.histogram2de outliers to geographic space locations

def hist_to_map_space(self):
    """
    unpacks histogram bins onto their contributing map locations (self.outliers_hist).
    Data type is a list of x,y coordinate pairs
    """
    hist_outliers = np.zeros(self.mat_shape, dtype = int)
    # hist_outliers = np.zeros(self.mat_shape, dtype = int)  #REMOVED
    # idarr = np.where(pd.notna(self.bin_loc))  # id where bin array has a tuple of locations
    idarr = np.where(self.outliers_hist_space)
    for i in range(len(idarr[0])):  # iterate through all bins with tuples
        loc_tuple = self.bin_loc[idarr[0][i], idarr[1][i]]
        for j in range(len(loc_tuple)):  # iterate through each tuple within each bin
            pair = loc_tuple[j]
            hist_outliers[pair[0], pair[1]] = 1
    print(type(hist_outliers))
    print(type(hist_outliers[0]))
    self.outliers_hist = hist_outliers  # unpacked map-space locations of 
