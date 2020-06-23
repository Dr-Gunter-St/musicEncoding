import wawOpener as ww
import os
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

processor = ww.Processor()
totoOriginal = processor.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\originals" + os.sep + "africa-toto.wav", 'rb')
totoCover = processor.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\covers" + os.sep + "toto-metal-cover-cut.wav", 'rb')
print(totoOriginal.getparams()) #(nchannels, sampwidth, framerate, nframes, comptype, compname)
print(totoCover.getparams()) #(nchannels, sampwidth, framerate, nframes, comptype, compname)

totalFrames = min(totoOriginal.getnframes(), totoCover.getnframes())

#IMPORTANT
"""
frame = totoOriginal.readframes(1)
print(frame)
int_values = [x for x in frame]
print(int_values)
print(bytes(int_values))
#"""

#encoding_dim = totoOriginal.getsampwidth()
#input_img = Input(shape=(784,))

regularEncoder = processor.getEncoder(400, 16)

regularEncoder.compile(optimizer='Adadelta', loss='binary_crossentropy')
totoInput = processor.getTrainingSet(totoOriginal, 400)
print(totoInput.shape)
print(totoInput[0].shape)
totoOutput = processor.getTrainingSet(totoCover, 400)

history = regularEncoder.fit(totoInput, totoInput, batch_size=20, epochs=40, verbose=1)