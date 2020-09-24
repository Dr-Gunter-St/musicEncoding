from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

class RNN:

    def __init__(self):
        pass

    def getRNN(self):
        print('Build model...')
        model = Sequential(name='RNN-LSTM')
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.3, input_shape=(None, 1), return_sequences=True))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.3, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(32, activation='softmax'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])
        return model

    def fitRNN(self, model, x_train, y_train, validationSet):
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5,
                                verbose=1, mode='auto', restore_best_weights=True)
        print('Train...')
        history = model.fit(x_train, y_train, validation_data=validationSet,
                  callbacks=[monitor], verbose=1, epochs=100)
        model.save("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\heavyModels\\heavyModel1")

        history_dict = history.history
        print(history_dict.keys())

