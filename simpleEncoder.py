from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

class Encoder:

    def __init__(self):
        pass

    def getEncoder(self, inputDim, encodingDim):

        compressionFactor = int(inputDim / encodingDim)
        currentCompression = compressionFactor

        inputLayer = Input(shape=(inputDim,))
        encoded = inputLayer

        while currentCompression > 1:
            encoded = Dense(encodingDim * currentCompression, activation='relu')(encoded)
            currentCompression -= 0.5

        encoded = Dense(encodingDim, activation='relu')(encoded)
        decoded = Dense(encodingDim, activation='relu')(encoded)

        while currentCompression <= compressionFactor:
            decoded = Dense(encodingDim * currentCompression, activation='relu')(decoded)
            currentCompression += 0.5

        decoded = Dense(inputDim, activation='sigmoid')(decoded)

        autoencoder = Model(inputLayer, decoded)
        autoencoder.summary()

        return autoencoder

    def getFatEncoder(self, inputDim):
        inputLayer = Input(shape=(inputDim,))
        encoded = Dense(inputDim, activation='relu')(inputLayer)
        encoded = Dense(inputDim, activation='relu')(encoded)
        encoded = Dense(inputDim, activation='relu')(encoded)

        decoded = Dense(inputDim, activation='relu')(encoded)
        decoded = Dense(inputDim, activation='relu')(decoded)
        decoded = Dense(inputDim, activation='sigmoid')(decoded)

        autoencoder = Model(inputLayer, decoded)
        autoencoder.summary()

        return autoencoder