# -*- coding: utf-8 -*-
# Loading the Libraries
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from fnmatch import fnmatch

root = pathlib.Path().absolute()
pattern = "*.wav"
nomes_audios = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            nomes_audios.append(os.path.join(path, name))

for name in nomes_audios:
    samplerate, data = read(name)
    # Frame rate for the Audio
    print(name)
    
    # Duration of the audio in Seconds
    duration = len(data)/samplerate
    print("Duration of Audio in Seconds", duration)
    print("Duration of Audio in Minutes", duration/60)
    
    time = np.arange(0,duration,1/samplerate)
    
    # Plotting the Graph using Matplotli
    plt.plot(time,data)
    plt.xlabel('Tempo (s)')
    plt.title(str(name))
    plt.ylabel('Amplitude')
    plt.show()
