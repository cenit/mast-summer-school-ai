#!/usr/bin/env python3

import tensorflow as tf
import tflite_model_maker as mm
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import audio
from tflite_model_maker import audio_classifier
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import glob
import random

# from IPython.display import Audio, Image
from IPython import display
from scipy.io import wavfile

print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Maker Version: {mm.__version__}")

# @title [Run this] Util functions and data structures.

data_dir = './dataset/cries_dataset'

cry_code_to_name = {
    'hiccups': 'Hiccups',
    'cramps': 'Cramps or trapped wind',
    'hungerroot': 'Hunger/rooting',
}


test_files = os.path.abspath(os.path.join(data_dir, 'test/*/*.wav'))


def get_random_audio_file():
    test_list = glob.glob(test_files)
    random_audio_path = random.choice(test_list)
    print(f'Cry file!: {random_audio_path}')
    return random_audio_path


def show_cries_data(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path, 'rb')

    cry_code = audio_path.split('/')[-2]
    print(f'Cry type: {cry_code_to_name[cry_code]}')
    print(f'Cry code: {cry_code}')

    plttitle = f'{cry_code_to_name[cry_code]} ({cry_code})'
    plt.title(plttitle)
    plt.plot(audio_data)
    display.display(display.Audio(audio_data, rate=sample_rate))


print('functions and data structures created')

random_audio = get_random_audio_file()
show_cries_data(random_audio)

spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=True,
    frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

train_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(data_dir, 'train'), cache=True)
train_data, validation_data = train_data.split(0.8)
test_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(data_dir, 'test'), cache=True)

batch_size = 5
epochs = 100

print('Training the model')
model = audio_classifier.create(
    train_data,
    spec,
    validation_data,
    batch_size=batch_size,
    epochs=epochs)

print('Evaluating the model')
model.evaluate(test_data)

print('\nConfusion matrix: ')
print(model.confusion_matrix(test_data))
print('labels: ', test_data.index_to_label)

models_path = './cries_models'
print(f'Exporting the TFLite model to {models_path}')

model.export(models_path, tflite_filename='my_cries_model.tflite')

model.export(models_path, export_format=[
             mm.ExportFormat.SAVED_MODEL, mm.ExportFormat.LABEL])
