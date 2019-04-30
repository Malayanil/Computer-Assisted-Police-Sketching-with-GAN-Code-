########################
# NPY DATA MERGER FILE #
########################

# Importing the required libraries
import matplotlib.pyplot as plt 
import numpy as np
import glob
import os, sys

# Assigning paths to path-variables
fpath ="D:\Everything Else\Major Project\Code"
npyfilespath ="D:\Everything Else\Major Project\Code"   

# Changing directory
os.chdir(npyfilespath)

# Merging the files 
npfiles= glob.glob("*.npy")
npfiles.sort()
all_arrays = []
for i, npfile in enumerate(npfiles):
    all_arrays.append(np.load(os.path.join(npyfilespath, npfile)))

# Saving the merged data file
np.save(fpath, np.concatenate(all_arrays))