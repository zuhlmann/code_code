import numpy as np
import math

#plottables

class Plotables:
    "this does some stuff"
    def cb_readable(self,array,num_ticks):
        ''' this just gives tick marks plotting '''
        mn = np.amin(array)
        mx = np.amax(array)
        #below if elif ensures colorbar tick values are readable.  needs some work.
        if mx-mn<0.1:
            self.cb_range = np.array([mn, mx])
        elif 0.9*mx - 1.1*mn < 100*.1:
            cb_range = np.linspace(mn*1.1, mx*0.9, num_ticks)
            self.cb_range = self.range_cust(cb_range, 1)
        elif 0.9*mx - 1.1*mn < 100*.5:
            cb_range = np.linspace(mn*1.1, mx*0.9, num_ticks)
            self.cb_range = self.range_cust(cb_range, 1)
        elif 0.9*mx - 1.1*mn < 100*1:
            cb_range = np.linspace(mn*1.1, mx*0.9, num_ticks)
            self.cb_range = self.range_cust(cb_range, 0)
        elif 0.9*mx - 1.1*mn < 100*10:
            cb_range = np.linspace(mn*1.1, mx*0.9, num_ticks)
            self.cb_range = self.range_cust(cb_range, -1)
    # rd = input('min = {:.2} max = {:2}  \n enter rounding precision as integer  \n ex) -1 = nearest 10, 2 = two decimal places: \n '.format(mn,mx))
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
                imgs = imgs - 1  #countergcdf
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

    def range_cust(self, np_obj, rd):
        # ZRU  5/20/19  This needs work.  Just ensure min rounds up and max rounds down
        np_objT = np_obj   #temp object
        for i, j in enumerate(np_obj):
            if rd > 0:
                np_objT[i] = round(j, rd)
            elif rd <= 0:
                np_objT[i] = int(round(j, rd))  # reduce numbers to display in colorbar
        return np_objT
