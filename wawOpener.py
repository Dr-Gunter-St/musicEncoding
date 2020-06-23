import wave
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
import numpy as np


class Processor():

    def __init__(self):
        pass

    def openWave(self, filename, mode):
        file = wave.open(filename, mode)
        return file

    def getEncoder(self, inputDim, compressionFactor):
        encodingDim = int(inputDim / compressionFactor)

        inputLayer = Input(shape=(inputDim,))
        encoded = Dense(encodingDim * 4, activation='relu')(inputLayer)
        encoded = Dense(encodingDim * 2, activation='relu')(encoded)
        encoded = Dense(encodingDim, activation='relu')(encoded)

        decoded = Dense(encodingDim * 2, activation='relu')(encoded)
        decoded = Dense(encodingDim * 4, activation='relu')(decoded)
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

    def getTrainingSet(self, readFile, inputDim):
        # prepare array of right shape
        totalFrames = readFile.getnframes()
        sampwidth = readFile.getsampwidth()
        numberOfSets = (totalFrames*sampwidth) / inputDim

        if numberOfSets % 1 > 0:
            numberOfSets = int(numberOfSets) + 1
        else:
            numberOfSets = int(numberOfSets)

        x_train = np.empty((numberOfSets, inputDim))

        if  inputDim/sampwidth % 1 > 0:
            raise ValueError("Input dimension has to be divisible by sampwidth")

        # miscellaneous vars
        currentPos = 0
        currentSet = 0
        readFile.setpos(currentPos)

        while currentPos*4 + inputDim < totalFrames*4:
            frame = readFile.readframes(int(inputDim/sampwidth))
            int_values = [x for x in frame]
            x_train[currentSet] = np.array(int_values)
            currentPos+=int(inputDim/sampwidth)
            currentSet+=1

        if currentPos < totalFrames:
            frame = readFile.readframes(totalFrames - currentPos)
            int_values = [x for x in frame]
            while len(int_values) < inputDim:
                int_values.append(0)
            x_train[currentSet] = np.array(int_values)

        return self.normalize(x_train)

    def normalize(self, inputSet):
        return inputSet.astype('float32') / 255.


    def deNormilize(self, outputSet):
        # DENORMALIZE
        return 0


