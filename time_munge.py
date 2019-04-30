import pandas as pd
from matplotlib import pyplot as plt
import datetime as dtt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import h5py

class DispTimeSeries:
    """let's see"""
    def __init__(self, st_date, stop_date):
        self.h5py_obj = h5py.File('../../projects/learning/brb_z/data/output/brb/devel/wy2017/awsm_paper/data/data4368_6528/smrfOutputs/precip.nc', 'r')
        self.data = self.h5py_obj["precip"]
        self.start = st_date  #get from rcdump in terminal
        self.stop = stop_date
        self.obs_days = (self.data.shape[0]-1)/24  #Note: if using input data, it's on daily timesteps not hourly
    # Print stuff about dataframe
    def find_range(self):
        print(self.data.index[[0,-1]])
    def subset(self):
        start = pd.to_datetime(self.start)
        self.start = start.strftime('%Y-%m-%d %H:%M:%S')
        stop = pd.to_datetime(self.stop)
        self.stop = stop.strftime('%Y-%m-%d %H:%M:%S')

    def print_col_names(self):
        col = self.data.columns  #column names
        str_out = '\n'.join([col[i] for i in range(len(col))])
        str_out = "COLUMN NAMES: \n\n" + (str_out)
        print("\n")
        print(str_out)
        print('-'*50)
        print("NUMBER OF ENTRIES: \n", len(data))

    def show_dates(self):
        print("-"*50)
        print("RESULTS: subsetted data from date range\n")
        print(self.sub)

    def plot_Z(self):
        plt.plot(self.sub.RME_176)
        plt.show()


# data = pd.read_csv('air_temp.csv', index_col = "date_time")
# test = DispTimeSeries(data, '1998-01-01', '1998-01-05')
# test.subset()
# test.show_dates()
# test.find_range()
# test.plot_Z()
