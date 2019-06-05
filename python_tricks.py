
""" INDEXING """

# 1) indexing a dataframe with pandas
dfi.iloc[1:4,3:7]  #subsets rows 1-4 and col 3-7
dfi.iloc[[1,2,4],[3,4,7]]  #subsets with actual rws cls
data.iloc[[0,-1]]  #subset the first and last row

# 2) indexing h5py Files '''
mask_file = h5py.File('../brb/topo/topo.nc', 'r')
print(list(mask_file.keys()))
mask = mask_file['mask'][0:3,0:3]  #this will grab rows and cols 0 to 3

#3) indexing/subsetting 3D arrays
# sets to true all layers from row 3:150 and col 25:320
trim_mat[:, 3:150, 25:320] = True

''' DICTIONARIES and KEYS '''
test ={'array 1':np.array([1,1,3]), 'array 2':np.array([3,4,5])}
print(test.keys())
print(test['array 1'])

''' VARIOUS USEFUL FUNCTIONS '''
np.zeros_like() #basicly zeros(row, col)

#this is how ENUMERATE works
iter = np.random.randint(1,10,20)
print(list(enumerate(iter)))

#similiarly to keys; from python docs 5.6
questions = {'name', 'quest', 'favorite color'}
answers = {'lancelot', 'the holy grail', 'blue'}
for q, a in zip(questions, answers):
    print('What is your {0}?  It is {1}.'.format(q,a))

''' RECAST and COPY dtypes'''
# 1) recast np.array:
mat = np.array([1,2,3])
mat = mat.astype('float64') # creates a copy of object, cast to new datatype
   #NOT:
mat = np.array([1,2,3])
mat.astype('float64')


''' PRINT the command along with output in terminal '''

str = 'cmocean.cm.matter_r(np.linspace(0., 1, 127))'
print(str)
print(eval(str))  #eval for 2 + 2 stuff, exec() for ?
print(exec(str))

''' CREATING MASKS '''
# 1) MAKING a Boolean Matrix (adapted from getCDF.trim_to_NA_extent )
# make a boolean false matrix
trim_mat = np.full((12, self.nrows, self.ncols), False, dtype = bool)
# add True where we want extent of clip
trim_mat[:, mnr:mxr, mnc:mxc] = True

# 2) BOOLEAN from 2D array - similiar to previous (image = np array)
# 1) make inputs = -9999 value.
booli = np.where(image!=-9999)
b1, b2 = min(booli[0]), max(booli[0])
b3, b4 = min(booli[1]), max(booli[1])
image_clip = image[b1:b2,b3:b4]

# Also selecting conditionally
(nc_array == 42)   #yields a boolean matrix where condition is met
(nc_array == 42).sum()   # this finds the total number of elemnts meeting condition

''' FORMATTING '''
print('the basin area (masked) = {:.1f} acres' .format(gcdf_obj.basin_area))
