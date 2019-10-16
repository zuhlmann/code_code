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


fig, ax1 = plt.subplots()

color = ['r', 'g', 'y', 'b']
labels = ['raw change(cm)', 'normalized change']
ax1.set_xlabel('elevation (m)')
ax1.set_ylabel(labels[0])
ax1.plot(elevation, upper, 'r-')
ax1.plot(elevation, lower, 'r-')
ax1.tick_params(axis = 'y', labelcolor = 'r')

ax2 = ax1.twinx()

ax2.set_ylabel(labels[1])
ax2.plot(elevation, upper_norm, 'k-')
# axe2.tick_params(axis = 'y', labelcolor = color[2])
ax2.plot(elevation, lower_norm, 'k-')
ax2.tick_params(axis = 'x', labelcolor = 'k')

fig.tight_layout()
plt.show()
