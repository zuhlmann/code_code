import numpy as np
import matplotlib.pyplot as plt
import math

''' this is megans plotting script '''

class PlotFct():
    def __init__(self):
        pass
    def plot_simple(self):
        self.num_subs = self.mat.shape[0]
        if self.num_subs == 1:
          fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (6,8), dpi = 180)  # 1 image
        elif self.num_subs == 2:
          fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (6,8), dpi = 180)  # 2 images
        else:  # 2 +
          fig, axs = plt.subplots(nrows = math.ceil(self.num_subs/3), ncols = 3, figsize = (6,8), dpi = 180)

        axs = axs.ravel()
        mat = self.mat
        for j, axo in enumerate(axs):
            try:
                axo.imshow(mat[j,:,:])
            except IndexError:
                axo.axis("off")
        plt.show()

    def get_files(self, fp):
        file_list = os.listdir(fp)
        # open file_list[0] -->  = tmp
        mat = np.empty(len(file_list), tmp.shape[0],tmp.shape[1]))
        for id, file_list in enumerate(file_list):
            mat[id] = open_file_fctn(file_list)
        self.mat = mat
