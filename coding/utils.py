#!/usr/bin/python

import numpy as np
import cv2
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def parse_csv(dirname, val_split=None, correction=None):
	'''
	Parse the csc file and return the image file names & steering angles
	additionally, adds a correction factor to left and right images if defined
	'''
	lines=[]
	with open(dirname+'/driving_log.csv','r') as csvfile:
		reader=csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	images=[]
	measurements=[]
	for line in lines:
		source_path=line[0]
		filename=dirname+'/IMG/'+source_path.split('/')[-1]
		images.append(filename)
		measurements.append(float(line[3]))
		if correction: # if correction factor is defined - read left and right
			left_src=line[1]
			images.append(dirname+'/IMG/'+left_src.split('/')[-1])
			measurements.append(float(line[3])+correction) # left is negative
			right_src=line[2]
			images.append(dirname+'/IMG/'+right_src.split('/')[-1])
			measurements.append(float(line[3])-correction) # right is positive
	assert len(images)==len(measurements), "Number of images don't match measurements"
	# Make test train split
	if val_split:
		X_train, X_valid, y_train, y_valid = train_test_split(np.array(images), np.array(measurements), test_size=val_split)
		return (X_train, y_train), (X_valid, y_valid)
	else:
		return (images, measurements)

def simple_generator(samples,batch_size=32):
	'''
	Simple generator: yeilds images and labels

	'''
	# Samples are (images, measurements)
	filenames=samples[0]
	measurements=samples[1]

	len_samples=len(filenames)
	while True:# Continuously yield data
		shuffle(filenames,measurements)
		for offset in range(0, len_samples,batch_size):
			batch_filenames=filenames[offset:offset+batch_size]
			batch_measurements=measurements[offset:offset+batch_size]
			images=[]
			angles=[]
			for img, ang in zip(batch_filenames,batch_measurements):
				images.append(cv2.imread(img))
				angles.append(ang)
			X_train=np.array(images)
			y_train=np.array(angles)
			yield shuffle(X_train,y_train)

def plot_history(hist):
	''' 
	Plots the history object
	'''
	plt.plot(hist['loss'])
	plt.plot(hist['val_loss'])
	plt.title('Model Training Display')
	plt.ylabel('Mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training_set','validation_set'], loc='upper right')
	plt.show()