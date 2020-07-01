import wawOpener as ww
import os
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adadelta

tf.compat.v1.disable_eager_execution()
processor = ww.Processor()
totoOriginal = processor.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\originals" + os.sep + "africa-toto-16bit.wav", 'rb')
totoCover = processor.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\covers" + os.sep + "toto-metal-cover-cut-16bit.wav", 'rb')
print(totoOriginal.getparams()) #(nchannels, sampwidth, framerate, nframes, comptype, compname)
print(totoCover.getparams()) #(nchannels, sampwidth, framerate, nframes, comptype, compname)
totoCover.getfp()

totalFrames = min(totoOriginal.getnframes(), totoCover.getnframes())

processor.getAmplitudes(totoOriginal)
processor.getAmplitudes(totoCover)
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

"""
regularEncoder = processor.getEncoder(40, 4)

optimizer = Adadelta(lr=0.5)

regularEncoder.compile(optimizer, loss='binary_crossentropy')
totoInput = processor.getTrainingSet(totoOriginal, 40, totalFrames)
print(totoInput.shape)
print(totoInput[0].shape)
totoOutput = processor.getTrainingSet(totoCover, 40, totalFrames)

history = regularEncoder.fit(totoInput, totoOutput, batch_size=200, epochs=40, verbose=1)

newCover = processor.writeCover(totoInput, regularEncoder)
"""



totoOriginal.close()
totoCover.close()