#!/usr/bin/env python3

# import the packages

import os
import torchaudio
import IPython.display as ipd
import matplotlib.pyplot as plt

default_dir = os.getcwd()
folder = 'data'

print(f'Data directory will be: {default_dir}/{folder}')

if os.path.isdir(folder):
    print("Data folder exists.")
else:
    print("Creating folder.")
    os.mkdir(folder)

trainset_speechcommands = torchaudio.datasets.SPEECHCOMMANDS(f'./{folder}/', download=True)

print("Completed!")
