#!/usr/bin/python 


from utils import parse_csv, simple_generator, plot_history
from cnn_models import simple_model, nvidia_net

dataset_path='data_sample'
training_samples, validation_samples=parse_csv(dataset_path,val_split=0.2,correction=0.2)

train_generator=simple_generator(training_samples,batch_size=32)
validation_generator=simple_generator(validation_samples,batch_size=32)

model=nvidia_net()
print("Training with: {} samples".format(len(training_samples[0])))
hist_obj=model.fit_generator(train_generator, samples_per_epoch=len(training_samples[0]), 
	validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model.h5')
plot_name="initial_training.png"
plot_history(hist_obj.history,plot_name)
