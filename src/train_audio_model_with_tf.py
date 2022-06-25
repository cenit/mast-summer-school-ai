#!/usr/bin/env python3

import pathlib
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
import IPython.display as ipd
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K
import tensorflowjs as tfjs
import json

# %%
skip_downloading_dataset = True
skip_extracting_dataset = True
skip_creating_spectrograms = True
data_dir = './data/'

print(f'Tensorflow version: {tf. __version__}')


# Let's think about the digital representation of analog sound. How does sound get recorded anyway?! Just like with images we need to take our physical world and convert it to numbers or a digital representation for a computer to understand. For audio, a microphone is used to capture the sound and then it's converted from analog sound to digital sound by sampling at consistent intervals of time. This is called the `sample rate`. The higher the `sample rate` the higher the quality of the sound; however after a certain point the difference is not able to be detected by the human ear. The average sound sample rate is 48 kHz or 48,000 samples per second. The dataset we'll use in this module was sampled at 16kHz, so our data was sampled 16,000 times per second.
#
# When we sample audio, we measure the `amplitude` or how loud the audio is, at a particular sample rate. We can then take that information and represent the `signal` in graphical format as a `waveform`. Another way to look at the audio information is by analyzing its `frequency` or pitch, which represents the number of complete waves per time period -- the more waves, the higher the pitch! Also, audio can be recorded in different `channels`. For example stereo recordings have 2 channels, right and left.
#
# Now that we understand a bit about how we get our audio file, let's take a moment to understand how we might want to parse out a file. If you have longer audio files, you may want to split it out into `frames` or sections of the audio to be classified individually. For this dataset we don't need to split the data into frames because each sample is only one second and one word. Another processing step might be an `offset` which means the number of frames from the start of the file to begin data loading.
#

# %% [markdown]
# ## Get the speech commands dataset
if not skip_downloading_dataset:
    origin = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
else:
    origin = f'{data_dir}/speech_commands_v0.02.tar.gz'

if not skip_extracting_dataset:
    tf.keras.utils.get_file(
        'speech_commands_v0.02.tar.gz',
        origin=origin,
        extract=True,
        cache_dir='.', cache_subdir='data')

default_dir = os.getcwd()
os.chdir(data_dir)
print(f'Data directory will be: {os.getcwd()}')

def load_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(audio, axis=-1)
    return waveform, sample_rate


def load_audio_files(path: str, label: str):
    dataset = []
    walker = sorted(str(p) for p in glob.glob(os.path.join(path, '*.wav')))
    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)
        #speaker_id, utterance_number = speaker.split("_nohash_")
        #utterance_number = int(utterance_number)
        waveform, sample_rate = load_audio(file_path)
        #dataset.append([waveform, sample_rate, label, speaker_id, utterance_number])
        dataset.append([waveform, sample_rate, label])

    return dataset


# %%
labels = [name for name in os.listdir('.') if os.path.isdir(name)]
os.chdir(default_dir)
print(f'Total Labels: {len(labels)}')
print(f'Label Names: {labels}')

num_classes = len(labels)
trainset = [None] * num_classes

# %% [markdown]
# ## Convert the sound to tensor
# - Call the `load_audio_files` function for each class we are going to use, then print the length of the dataset.
for i in range(num_classes):
    trainset[i] = load_audio_files(
        f'{data_dir}/{labels[i]}/', f'{labels[i]}')
    print(f'Length of dataset {labels[i]}: {len(trainset[i])}')

# %%
# - Now let's grab an example item from each dataset. We can see the waveform, sample_rate, label, and id.
for i in range(num_classes):
    print(f'Class {labels[i]}:')
    print(f'Waveform: {trainset[i][0]}')
    print(f'Sample Rate: {trainset[i][1]}')
    print(f'Label: {trainset[i][2]}')
    #print(f'ID: {trainset[i][3]}')

# %%
# ### Spectrogram
#
# Next we will look at the `Spectrogram`. What is a spectrogram anyway?! A spectrogram allows you to visualize the amplitude as a function of frequency and time in the form of an image, where the 'x' axis represents time, the 'y' axis represents frequency, and the color represents the amplitude. This image is what we will use for our computer vision classification on our audio files.
#
# Here we look at two different ways to create the spectrogram from the waveform. First we want to make our waveforms all equal lengths so we will pad them with zeros. Then we apply to transforms [tf.signal.stft](https://www.tensorflow.org/api_docs/python/tf/signal/stft) and [tfio.audio.spectrogram](https://www.tensorflow.org/io/api_docs/python/tfio/audio/spectrogram?hl=da).

frame_length = 232
frame_step = 128
sample_rate = 16000

def get_spectrogram(waveform):
    # Padding for files with less than {sample_rate} samples
    if tf.shape(waveform)[0] < sample_rate:
        zero_padding = tf.zeros([sample_rate] - tf.shape(waveform), dtype=tf.float32)
        # Concatenate audio with padding so that all audio clips will be of the same length
        waveform = tf.cast(waveform, tf.float32)
        equal_length_waveform = tf.concat([waveform, zero_padding], 0)
    else:
        waveform = tf.cast(waveform, tf.float32)
        equal_length_waveform = waveform

    # Option 1: Use tfio to get the spectrogram
    spect = tfio.audio.spectrogram(
        input=equal_length_waveform, nfft=frame_length, window=frame_length, stride=frame_step)

    ## Option 2: Use tf.signal processing to get the Short-time Fourier transform (stft)
    #spectrogram = tf.signal.stft(
    #    equal_length_waveform, frame_length=frame_length, frame_step=frame_step)
    #spectrogram = tf.abs(spectrogram)

    #return spectrogram, spect
    return spect


# %%
# ## Save the spectrogram as an image
#
# We have broken down some of the ways to understand our audio data and different transformations we can use on our data. Now lets create the images we will use for classification.
#
# Below is a function to create the Spectrogram image for classification.

def create_images(dataset, label_dir):
    # make directory
    test_directory = f'./test/{label_dir}/'
    train_directory = f'./train/{label_dir}/'

    os.makedirs(test_directory, mode=0o777, exist_ok=True)
    os.makedirs(train_directory, mode=0o777, exist_ok=True)

    for i, data in enumerate(dataset):

        waveform = data[0]
        #spectrogram, spect = get_spectrogram(waveform)
        spect = get_spectrogram(waveform)

        # Split test and train images by 30%
        if i % 3 == 0:
            plt.imsave(
                f'{test_directory}/spec_img{i}.png', spect.numpy(), cmap='gray')
                #f'./test/{label_dir}/spec_img{i}.png', spectrogram.numpy(), cmap='gray')
        else:
            plt.imsave(
                f'{train_directory}/spec_img{i}.png', spect.numpy(), cmap='gray')
                #f'./train/{label_dir}/spec_img{i}.png', spectrogram.numpy(), cmap='gray')


if not skip_creating_spectrograms:
    for i in range(num_classes):
        print(
            f'Creating spectrogram for class {labels[i]} (warning, it might take a long time!)')
        create_images(trainset[i], labels[i])

# %% [markdown]
# Now that we have created the spectrogram images it's time to build the computer vision model
#
# Here we provide the path to our image data and use [tf.keras.preprocessing.image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory) to load the images into tensors.
#
# This method is doing a lot for us. Lets take a look at a few of the params:
# - `labels='inferred'`: The labels are created based on folder directory names.
# - `image_size=(img_width, img_height)`: resizes the image
# - `validation_split=0.2, subset='validation'`: create validation dataset

train_directory = './train/'
test_directory = './test/'
img_height = frame_length
img_width = 43


print('Preprocessing train dataset')
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory, labels='inferred', color_mode="grayscale", label_mode='int', image_size=(img_width, img_height), seed=123,
    validation_split=0.2, subset='validation')

print('Preprocessing test dataset')
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory, labels='inferred', color_mode="grayscale", label_mode='int', image_size=(img_width, img_height),
    validation_split=None, subset=None)

class_names = train_ds.class_names
print(class_names)

# %% [markdown]
# ## Create the model
#
# We are ready to create the Convolution Neural Network for the computer vision model to process the spectrogram images.
#
# To construct the linear layers we use the [tf.keras.Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) and pass in a list with each layer. Read more about the layers [here](https://www.tensorflow.org/api_docs/python/tf/keras/layers).

rescale = tf.keras.layers.Rescaling(
    scale=1.0/255, input_shape=(img_height, img_width, 3))
train_ds = train_ds.map(lambda image, label: (rescale(image), label))
test_ds = test_ds.map(lambda image, label: (rescale(image), label))

model = tf.keras.Sequential([
    #tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])


# %% [markdown]
# - Set the `learning_rate`, loss function `loss_fn`, `optimizer` and `metrics`.

learning_rate = 0.125
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate)
metrics = ['accuracy']
print('Compiling model')
model.compile(optimizer, loss_fn, metrics)

# %% [markdown]
# ## Train the model
# Set the epochs
epochs = 15
print('\nGetting ready to train.\n')
if os.name == 'nt':
    print('Warning, if you get an error about missing zlibwapi.dll, please download it from http://www.winimage.com/zLibDll/zlib123dllx64.zip\n')
print('\nFitting (warning, it might take a long time!)')
# Train the model.
history = model.fit(train_ds, epochs=epochs)
print('\nModel summary:')
model.summary()

# %% [markdown]
#  ## Test the model
#
# Awesome! You should have got somewhere between a 93%-95% accuracy by the 15th epoch. Here we grab a batch from our test data and see how the model performs on the test data.

# %%
correct = 0
batch_size = 0
for batch_num, (X, Y) in enumerate(test_ds):
    batch_size = len(Y)
    pred = model.predict(X)
    for i in range(batch_size):
        predicted = np.argmax(pred[i], axis=-1)
        actual = Y[i]
        #print(f'predicted {predicted}, actual {actual}')
        if predicted == actual:
            correct += 1
    break

print(f'Number correct: {correct} out of {batch_size}')
print(f'Accuracy {correct / batch_size}')

print('\nExporting model')
model.save("saved_model_pb")

print('\nExporting model in tflite format')
input_model = "saved_model_pb"
output_model_lite = "output.tflite"
output_folder_tfjs = "tfjs"
converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
tflite_quant_model = converter.convert()
with open(output_model_lite, 'wb') as o_:
    o_.write(tflite_quant_model)

print('\nExporting model in tensorflow.js format')
tfjs.converters.save_keras_model(model, output_folder_tfjs)
metadata = {
    'tfjsSpeechCommandsVersion': '0.4.0',
    'modelName': 'TMv2',
    'timeStamp': '2022-06-22T08:44:50.198Z',
    'wordLabels': labels
}
metadata_json_path = os.path.join(output_folder_tfjs, 'metadata.json')
json.dump(metadata, open(metadata_json_path, 'wt'))
print('\nSaved model metadata at: %s' % metadata_json_path)
