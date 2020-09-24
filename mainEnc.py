import wawProcessor as ww
import simpleEncoder as se
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizers import Adadelta

tf.compat.v1.disable_eager_execution()
processor = ww.Processor()
enc = se.Encoder()
totoOriginal = processor.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\originals" + os.sep + "africa-toto-8bit.wav", 'rb')
totoCover = processor.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\covers" + os.sep + "toto-metal-cover-cut-8bit.wav", 'rb')

print(totoOriginal.getparams()) #(nchannels, sampwidth, framerate, nframes, comptype, compname)
print(totoCover.getparams()) #(nchannels, sampwidth, framerate, nframes, comptype, compname)
totalFrames = min(totoOriginal.getnframes(), totoCover.getnframes())

processor.getAmplitudes(totoOriginal)
maxAmpl, minAmpl = processor.getAmplitudes(totoCover)
amplitude = maxAmpl - minAmpl

totoOriginal.close()
totoCover.close()


totoOriginal = processor.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\originals" + os.sep + "africa-toto-8bit.wav", 'rb')
totoCover = processor.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\covers" + os.sep + "toto-metal-cover-cut-8bit.wav", 'rb')


# Distribution of values
processor.drawDensity(totoOriginal, amplitude, 'Original song\'s amplitude values distribution')
processor.drawDensity(totoCover, amplitude, 'Cover song\'s amplitude values distribution')

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

regularEncoder.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])
totoInput = processor.getTrainingSet(totoOriginal, 40, normalize=True, normalization_factor=float(amplitude))
print(totoInput.shape)
print(totoInput[0].shape)
totoOutput = processor.getTrainingSet(totoCover, 40, normalize=True, normalization_factor=float(amplitude))

x_train, y_train, x_test, y_test = processor.getValidationSet(totoInput, totoOutput)

history = regularEncoder.fit(x_train, y_train, batch_size=2000, epochs=5, validation_data=(x_test, y_test))

newCover = processor.writeCover(totoInput, regularEncoder, de_normalize=True, de_normalization_factor=amplitude)

history_dict = history.history
print(history_dict.keys())

plt.figure(1)
plt.title('Model training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history_dict['accuracy'], label='Training accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()
plt.figure(2)
plt.title('Model training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history_dict['loss'], label='Training loss')
plt.plot(history_dict['val_loss'], label='Validation loss')
plt.legend()
plt.show()


totoOriginal.close()
totoCover.close()
