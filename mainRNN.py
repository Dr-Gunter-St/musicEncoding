import wawProcessor as ww
import rnn as r

import os
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adadelta

processor = ww.Processor()
RNN = r.RNN()
model = RNN.getRNN()
totoOriginal = processor.openWave(
    "", 'rb')
totoCover = processor.openWave(
    "", 'rb')

x_train, y_train = processor.getRNNTrainSequences(totoOriginal, totoCover, 50)
# print(x_train.shape)
# print(y_train.shape)

x_train, y_train, x_test, y_test = processor.getValidationSet(x_train, y_train)

RNN.fitRNN(model, x_train, y_train, (x_test, y_test))

totoOriginal.close()
totoCover.close()
