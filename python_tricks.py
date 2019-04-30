
""" Indexing """

# INDEXING a dataframe with pandas
dfi.iloc[1:4,3:7]  #subsets rows 1-4 and col 3-7
dfi.iloc[[1,2,4],[3,4,7]]  #subsets with actual rws cls
data.iloc[[0,-1]]  #subset the first and last row

# INDEXING h5py Files
mask_file = h5py.File('../brb/topo/topo.nc', 'r')
print(list(mask_file.keys()))
mask = mask_file['mask'][0:3,0:3]  #this will grab rows and cols 0 to 3

''' Dictionaries and Keys '''
test ={'array 1':np.array([1,1,3]), 'array 2':np.array([3,4,5])}
print(test.keys())
print(test['array 1'])

#similiarly to keys; from python docs 5.6
questions = {'name', 'quest', 'favorite color'}
answers = {'lancelot', 'the holy grail', 'blue'}
for q, a in zip(questions, answers):
    print('What is your {0}?  It is {1}.'.format(q,a))

 ''' Various Functions '''

np.zeros_like() #basicly zeros(row, col)

#this is how ENUMERATE works
iter = np.random.randint(1,10,20)
print(list(enumerate(iter)))

# numpy astype
# numpy astype creates a copy of object, cast to new datatype.  So do this:
mat = np.array([1,2,3])
mat = mat.astype('float64')
   #NOT:
mat = np.array([1,2,3])
mat.astype('float64')

# Add to Plotting/Formatting class
#image is an np array
# make inputs = -9999 value.
# booli = np.where(image!=-9999)
# b1, b2 = min(booli[0]), max(booli[0])
# b3, b4 = min(booli[1]), max(booli[1])
# image_clip = image[b1:b2,b3:b4]
