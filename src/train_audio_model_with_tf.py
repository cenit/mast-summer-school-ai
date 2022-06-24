# ## Load the dataset
# Here we import the packages and create a `load_audio` and `load_audio_files` function to load audio files from a specified path into a dataset.

# %%
import os
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
import IPython.display as ipd
import numpy as np

# %%
def load_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(audio, axis=-1)
    return waveform, sample_rate

def load_audio_files(path: str, label:str):

    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)
        speaker_id, utterance_number = speaker.split("_nohash_")
        utterance_number = int(utterance_number)

        # Load audio
        waveform, sample_rate = load_audio(file_path)
        dataset.append([waveform, sample_rate, label, speaker_id, utterance_number])

    return dataset

# %%
# - Call the `load_audio_files` function for each class we are going to use, then print the length of the dataset.

# %%
trainset_speechcommands_yes = load_audio_files('./data/yes', 'yes')
trainset_speechcommands_no = load_audio_files('./data/no', 'no')

print(f'Length of yes dataset: {len(trainset_speechcommands_yes)}')
print(f'Length of no dataset: {len(trainset_speechcommands_no)}')

# %%
# - Now let's grab an example item from each dataset. We can see the waveform, sample_rate, label, and id.

# %%
yes_waveform = trainset_speechcommands_yes[0][0]
yes_sample_rate = trainset_speechcommands_yes[0][1]
print(f'Yes Waveform: {yes_waveform}')
print(f'Yes Sample Rate: {yes_sample_rate}')
print(f'Yes Label: {trainset_speechcommands_yes[0][2]}')
print(f'Yes ID: {trainset_speechcommands_yes[0][3]}')

no_waveform = trainset_speechcommands_no[0][0]
no_sample_rate = trainset_speechcommands_no[0][1]
print(f'No Waveform: {no_waveform}')
print(f'No Sample Rate: {no_sample_rate}')
print(f'No Label: {trainset_speechcommands_no[0][2]}')
print(f'No ID: {trainset_speechcommands_no[0][3]}')

# %%
# ## Transform and visualize
# Our data is ready

# ### Spectrogram
#
# Next we will look at the `Spectrogram`. What is a spectrogram anyway?! A spectrogram allows you to visualize the amplitude as a function of frequency and time in the form of an image, where the 'x' axis represents time, the 'y' axis represents frequency, and the color represents the amplitude. This image is what we will use for our computer vision classification on our audio files.
#
# Here we look at two different ways to create the spectrogram from the waveform. First we want to make our waveforms all equal lengths so we will pad them with zeros. Then we apply to transforms [tf.signal.stft](https://www.tensorflow.org/api_docs/python/tf/signal/stft) and [tfio.audio.spectrogram](https://www.tensorflow.org/io/api_docs/python/tfio/audio/spectrogram?hl=da).
#

# %%
def get_spectrogram(waveform):

    frame_length = 255
    frame_step = 128
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length_waveform = tf.concat([waveform, zero_padding], 0)

    # Option 1: Use tfio to get the spectrogram
    spect = tfio.audio.spectrogram(input=equal_length_waveform, nfft=frame_length, window=frame_length, stride=frame_step)

    # Option 2: Use tf.signal processing to get the Short-time Fourier transform (stft)
    spectrogram = tf.signal.stft(equal_length_waveform, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)

    return spectrogram, spect


spectrogram, spect = get_spectrogram(yes_waveform)

print('Label:', 'yes')
print('Waveform shape:', yes_waveform.shape)
print('Spectrogram shape:', spectrogram.shape)
print('Spect shape:', spect.shape)

# %%
# ## Save the spectrogram as an image
#
# We have broken down some of the ways to understand our audio data and different transformations we can use on our data. Now lets create the images we will use for classification.
#
# Below is a function to create the Spectrogram image for classification.

# %%
def create_images(dataset, label_dir):
    # make directory
    test_directory = f'./data/test/{label_dir}/'
    train_directory = f'./data/train/{label_dir}/'

    os.makedirs(test_directory, mode=0o777, exist_ok=True)
    os.makedirs(train_directory, mode=0o777, exist_ok=True)

    for i, data in enumerate(dataset):

        waveform = data[0]
        spectrogram, spect = get_spectrogram(waveform)

        # Split test and train images by 30%
        if i % 3 == 0:
            plt.imsave(f'./data/test/{label_dir}/spec_img{i}.png', spectrogram.numpy(), cmap='gray')
        else:
            plt.imsave(f'./data/train/{label_dir}/spec_img{i}.png', spectrogram.numpy(), cmap='gray')




# %%
create_images(trainset_speechcommands_yes, 'yes')
create_images(trainset_speechcommands_no, 'no')

# %% [markdown]
# We now have our audio as spectrogram images and are ready to build the model!


# %% [markdown]
# Now that we have created the spectrogram images it's time to build the computer vision model. If you are following along with the learning path then you already created a computer vision model in the second module in this path.
# Like always we first import the packages we need to build the model.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
import tensorflow_io as tfio
import IPython.display as ipd

# %% [markdown]
# ## Load Spectrogram images into a dataset for training
#
# Here we provide the path to our image data and use [tf.keras.preprocessing.image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory) to load the images into tensors.
#
# This method is doing a lot for us. Lets take a look at a few of the params:
# - `labels='inferred'`: The labels are created based on folder directory names.
# - `image_size=(256, 256)`: resizes the image
# - `validation_split=0.2, subset='validation'`: create validation dataset

# %%
train_directory = './data/train/'
test_directory = './data/test/'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory, labels='inferred', label_mode='int', image_size=(256, 256), seed=123,
    validation_split=0.2, subset='validation')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory, labels='inferred', label_mode='int', image_size=(256, 256),
    validation_split=None, subset=None)

class_names = train_ds.class_names
print(class_names)

# %% [markdown]
# ## Create the model
#
# We are ready to create the Convolution Neural Network for the computer vision model to process the spectogram images.
#
# To construct the linear layers we use the [tf.keras.Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) and pass in a list with each layer. Read more about the layers [here](https://www.tensorflow.org/api_docs/python/tf/keras/layers).

# %%
num_classes = 2
img_height = 256
img_width = 256

model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
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

# %%
learning_rate = 0.125

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate)
metrics = ['accuracy']
model.compile(optimizer, loss_fn, metrics)

# %% [markdown]
# ## Train the model

# %%
# Set the epocks
epochs = 15
print('\nFitting:')

# Train the model.
history = model.fit(train_ds, epochs=epochs)

# %%
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


import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K
import tensorflow as tf

model.save("saved_model_pb")
input_model = "saved_model_pb"
output_model = "./models/output.tflite"

#to tensorflow lite
converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
tflite_quant_model = converter.convert()
with open(output_model, 'wb') as o_:
    o_.write(tflite_quant_model)
