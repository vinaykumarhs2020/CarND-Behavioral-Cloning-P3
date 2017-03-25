#!/usr/bin/python 
# Models used for behavioral cloning project

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

def simple_model():
	model = Sequential()
	model.add(Flatten(input_shape=(160,320,3)))
	model.add(Dense(1))
	model.compile(loss='mse',optimizer='adam')
	return model
