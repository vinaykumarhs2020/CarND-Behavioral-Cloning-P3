#!/usr/bin/python 
# Script to retrain the network

from utils import parse_csv, simple_generator, plot_history
from cnn_models import simple_model, nvidia_net
from keras.models import load_model
from keras.utils import plot_model


# New dataset path
dataset_path='additional_data/1'
training_samples, validation_samples=parse_csv(dataset_path,val_split=0.2,correction=0.2)

train_generator=simple_generator(training_samples,batch_size=32)
validation_generator=simple_generator(validation_samples,batch_size=32)

# Load the pretrained model
pretrained_model="model.h5"
model=load_model(pretrained_model)
print("Re-training with: {} samples".format(len(training_samples[0])))

# Trin the mode with new dataset generator
hist_obj=model.fit_generator(train_generator, samples_per_epoch=len(training_samples[0]), 
	validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=7)

retrained_model="model_re.h5"
plot_name="retrainig.png"
model_arch="model_arch.png"

# Save retrained model
model.save(retrained_model)
# Plot history of trainig epochs
plot_history(hist_obj.history,plot_name)
# Save model architecture
plot_model(model, to_file=model_arch)
