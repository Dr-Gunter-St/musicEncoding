from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class RNN:

    def __init__(self):
        pass

    def getRNN(self):
        print('Build model...')
        model = Sequential()
        model.add(LSTM(64, dropout=0.0, recurrent_dropout=0.0, input_shape=(None, 1)))
        # model.add(LSTM(32, dropout=0.0, recurrent_dropout=0.0, input_shape=(None, 1)))
        model.add(Dense(32))
        # model.add(Dense(16))
        model.add(Dense(1))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adadelta')
        return model

    def fitRNN(self, model, x_train, y_train, validationSet):
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5,
                                verbose=1, mode='auto', restore_best_weights=True)
        print('Train...')
        model.fit(x_train, y_train, validation_data=validationSet,
                  callbacks=[monitor], verbose=1, epochs=100)
        model.save("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\heavyModels\\heavyModel")

