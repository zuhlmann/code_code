

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------- OPTIONS FOR ADDING SUBPLOTS------------
#1) Gridspec to customize subplot sizes.  Just like that r plot package that specified size and location
# Here is a link:
#  https://matplotlib.org/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2:, :])

#2) or use sub_plot
# https://matplotlib.org/gallery/specialty_plots/mri_with_eeg.html#sphx-glr-gallery-specialty-plots-mri-with-eeg-py
fig = plt.figure("MRI_with_EEG")
ax0 = fig.add_subplot(2, 2, 1)
ax1 = fig.add_subplot(2, 2, 2)
ax2 = fig.add_subplot(2, 1, 2)

# definition to make cbar ticks
def cb_readable(array, num_ticks, decim):
    mn = np.amin(array)
    mx = np.amax(array)
    cb_range = np.linspace(mn*0.105, mx*0.95, num_ticks)

print(np.array([0,0.1]))
