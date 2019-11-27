import pandas as pd
from matplotlib import pyplot as plt

file_in = '/home/zachuhlmann/projects/data/SanJoaquin/20170402_20170605/USCASJ20170402_20170605_raqc.csv'


def get_prop(total):
    prop = round(ct/sum(hyps.elevation_count),4)
    return(prop)

hyps = pd.read_csv(file_in)
col_names = hyps.columns.tolist()

# prop = [round(ct/sum(hyps.elevation_count),4) for ct in hyps.elevation_count]
total = sum(hyps.elevation_count)
hyps['elevation_count'] = hyps['elevation_count'].apply(lambda x: ((x/total) > 0.0001))
hyps.drop(hyps[~hyps.elevation_count].index, inplace = True)
# hyps.apply(get_prop(row, total), axis = 1)
print(hyps)

elevation = hyps[col_names[0]]
upper = hyps[col_names[1]]
lower = hyps[col_names[3]]
upper_norm = hyps[col_names[2]]
lower_norm = hyps[col_names[4]]


fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (8,10))

color = ['r', 'g', 'y', 'b']
labels = ['raw change(cm)', 'normalized change']
ax[0].set_xlabel('elevation (m)')
ax[0].set_ylabel(labels[0])
ax[0].plot(elevation, upper, 'r-o')
ax[0].plot(elevation, lower, 'k-o')
ax[0].tick_params(axis = 'y', labelcolor = 'k')
ax[0].fill_between(elevation, lower, upper, facecolor = 'b', alpha = 0.5)
range = max(upper) - min(lower)
ax[0].set_ylim(min(lower) - range * 0.2, \
                max(upper) + range * 0.2)
ax[0].legend(loc = 2)
ax[0].set_title('Snow Depth Change (cm) USCASJ20170402_to_20170605')

ax[1].set_ylabel(labels[1])
ax[0].set_xlabel('elevation (m)')
ax[1].plot(elevation, upper_norm, 'r-o')
# axe2.tick_params(axis = 'y', labelcolor = color[2])
ax[1].plot(elevation, lower_norm, 'k-o')
ax[1].tick_params(axis = 'x', labelcolor = 'k')
ax[1].fill_between(elevation, lower_norm, upper_norm, facecolor = 'b', alpha = 0.5)
range = max(upper_norm) - min(lower_norm)
ax[1].set_ylim(min(lower_norm) - range * 0.2, \
                max(upper_norm) + range * 0.2)
ax[1].legend(loc = 2)
ax[1].set_title('Normalized Snow Depth Change (cm) USCASJ20170402_to_20170605')


fig.tight_layout()
plt.savefig('USCASJ20170402_to_20170605_Hypsometry.png', dpi = 180)
