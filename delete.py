# playing with git
import numpy as np

one = np.ones((10,10))
arr = one * np.random.normal(0,1,(10,10))
id = np.random.randint(2, size = [10,10])
arr_check = arr * id
print(arr_check)
