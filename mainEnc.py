import wawProcessor as ww
import simpleEncoder as se
import os
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adadelta

tf.compat.v1.disable_eager_execution()
processor = ww.Processor()
enc = se.Encoder()
totoOriginal = processor.openWave("", 'rb')
totoCover = processor.openWave("", 'rb')

print(totoOriginal.getparams()) #(nchannels, sampwidth, framerate, nframes, comptype, compname)
print(totoCover.getparams()) #(nchannels, sampwidth, framerate, nframes, comptype, compname)
totalFrames = min(totoOriginal.getnframes(), totoCover.getnframes())

processor.getAmplitudes(totoOriginal)
maxAmpl, minAmpl = processor.getAmplitudes(totoCover)
amplitude = maxAmpl - minAmpl

totoOriginal.close()
totoCover.close()

totoOriginal = processor.openWave("", 'rb')
totoCover = processor.openWave("", 'rb')

# Distribution of values
#processor.drawDensity(totoOriginal, amplitude)
#processor.drawDensity(totoCover, amplitude)

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

regularEncoder = enc.getEncoder(40, 16)



optimizer = Adadelta(lr=2.0, rho=0.75)

regularEncoder.compile(optimizer, loss='mean_squared_error')
totoInput = processor.getTrainingSet(totoOriginal, 40, normalize=True, normalization_factor=float(amplitude))
print(totoInput.shape)
print(totoInput[0].shape)
totoOutput = processor.getTrainingSet(totoCover, 40, normalize=True, normalization_factor=float(amplitude))


history = regularEncoder.fit(totoInput, totoOutput, batch_size=2000, epochs=5, verbose=1)

newCover = processor.writeCover(totoInput, regularEncoder, de_normalize=True, de_normalization_factor=amplitude)




totoOriginal.close()
totoCover.close()
