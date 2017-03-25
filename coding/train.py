import csv
import cv2
import numpy as np
print(cv2.__file__)

lines=[]
with open('data_sample/driving_log.csv','rb') as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images=[]
measurements=[]
for line in lines:
	source_path=line[0]
	filename='data_sample/IMG/'+source_path.split('/')[-1]
	image=cv2.imread(filename)
	images.append(image)
	measurements.append(float(line[3]))
X_train = np.array(images)
y_train = np.array(measurements)

print(y_train.shape)
print(X_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3) ))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
