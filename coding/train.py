#!/usr/bin/python 


from utils import parse_csv, simple_generator
from cnn_models import simple_model

dataset_path='data_sample'
training_samples, validation_samples=parse_csv(dataset_path,val_split=0.2)

train_generator=simple_generator(training_samples,batch_size=32)
validation_generator=simple_generator(validation_samples,batch_size=32)

model=simple_model()
model.fit_generator(train_generator, samples_per_epoch=len(training_samples), 
	validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
