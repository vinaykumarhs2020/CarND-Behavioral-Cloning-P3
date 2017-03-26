#!/usr/bin/python 
# Models used for behavioral cloning project

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dense, Dropout, Cropping2D, Convolution2D

def simple_model():
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
	model.add(Flatten())
	model.add(Dense(1))
	model.compile(loss='mse',optimizer='adam')
	return model


def nvidia_net(dropout=0.1):
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
	model.add(Dropout(dropout))
	model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
	model.add(Dropout(dropout))
	model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
	model.add(Dropout(dropout))
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Dropout(dropout))
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	model.compile(loss='mse',optimizer='adam')
	return model

def mod_nvidia_net(dropout=0.15):
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(32,5,5,subsample=(2,2), activation='relu'))
	model.add(Dropout(dropout))
	model.add(Convolution2D(64,5,5,subsample=(2,2), activation='relu'))
	model.add(Dropout(dropout+0.05))
	model.add(Convolution2D(64,5,5,subsample=(2,2), activation='relu'))
	model.add(Dropout(dropout+0.05))
	model.add(Convolution2D(96,3,3, activation='relu'))
	model.add(Dropout(dropout+0.1))
	model.add(Convolution2D(96,3,3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	model.compile(loss='mse',optimizer='adam')
	return model
