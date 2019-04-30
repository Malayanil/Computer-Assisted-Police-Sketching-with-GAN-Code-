#######################
# DATA GENERATOR FILE #
#######################

# Importing required libraries 
import os, random, sys
import numpy as np
import cv2
from dutil import *

# Assigning values to variables
NUM_IMAGES = 680
SAMPLES_PER_IMG = 10
DOTS_PER_IMG = 60
IMAGE_W = 144
IMAGE_H = 192
IMAGE_DIR = 'PPMFiles'
NUM_SAMPLES = (NUM_IMAGES * 2 * SAMPLES_PER_IMG)

# Defining the function to resize images into 'n x n' pixels
def center_resize(img):
	assert(IMAGE_W == IMAGE_H)
	w, h = img.shape[0], img.shape[1]
	if w > h:
		x = (w-h)/2
		img = img[x:x+h,:]
	elif h > w:
		img = img[:,0:w]
	return cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation = cv2.INTER_LINEAR)

# Defining the function to return a resized image, using CV2
def ppm_resize(img):
	assert(img.shape[1] == 56)
	assert(img.shape[0] == 56)
	return cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation = cv2.INTER_LINEAR)
	
# Defining the function to return parameters to 'auto_canny' function of dutil.py
def rand_dots(img, sample_ix):
	sample_ratio = float(sample_ix) / SAMPLES_PER_IMG
	return auto_canny(img, sample_ratio)

# Assigning null values to 'x_data' and 'y_data' and 0 to 'ix', the index
x_data = np.empty((NUM_SAMPLES, 3, IMAGE_H, IMAGE_W), dtype=np.uint8)
y_data = np.empty((NUM_SAMPLES, 3, IMAGE_H, IMAGE_W), dtype=np.uint8)
ix = 0

# Loop to traverse the directories and read the files for data
for root, subdirs, files in os.walk(IMAGE_DIR):
	for file in files:
		path = root + "\\" + file
		if not path.endswith('.ppm'):
			continue
		img = cv2.imread(path)
		if img is None:
			assert(False)
		if len(img.shape) != 3 or img.shape[2] != 3:
			assert(False)
		if img.shape[0] < IMAGE_H or img.shape[1] < IMAGE_W:
			assert(False)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = ppm_resize(img)
		for i in range(SAMPLES_PER_IMG):
			y_data[ix] = np.transpose(img, (2, 0, 1))
			x_data[ix] = rand_dots(img, i)
			if ix < SAMPLES_PER_IMG*16:
				outimg = x_data[ix][0]
				cv2.imwrite('cargb' + str(ix) + '.png', outimg)
				print ( path )
			ix += 1
			y_data[ix] = np.flip(y_data[ix - 1], axis=2)
			x_data[ix] = np.flip(x_data[ix - 1], axis=2)
			ix += 1
			
		sys.stdout.write('\r')
		progress = ix * 100 / NUM_SAMPLES
		sys.stdout.write(str(progress) + "%")
		sys.stdout.flush()
		assert(ix <= NUM_SAMPLES)

# Saving the data for training the model
assert(ix == NUM_SAMPLES)
print ("\nSaving...")
np.save('x_data.npy', x_data)
np.save('y_data.npy', y_data)
print ("Done")
