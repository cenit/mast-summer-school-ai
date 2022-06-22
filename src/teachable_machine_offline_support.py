#!/usr/bin/env python3

# Imports
from tflite_support.task import audio
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

import os

default_dir = os.getcwd()
model_path = default_dir + '/soundclassifier_with_metadata.tflite'
print(f'Model in use: {model_path}')

# Initialization
base_options = core.BaseOptions(file_name=model_path)
classification_options = processor.ClassificationOptions(max_results=2)
options = audio.AudioClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = audio.AudioClassifier.create_from_options(options)

# Run inference
audio_path = default_dir + '/../res/Ensoniq-ZR-76-01-Dope-77.wav'
print(f'Audio for test will be: {audio_path}')
audio_file = audio.TensorAudio.create_from_wav_file(audio_path, classifier.required_input_buffer_size)
audio_result = classifier.classify(audio_file)
print(f'Class: {audio_result.classifications[0].categories[0].category_name}, probability: {audio_result.classifications[0].categories[0].score}')
