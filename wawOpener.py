import os
import random
import wave
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
import numpy as np


class Processor():

    nchannels = 1
    sampwidth = 1
    framerate = 1
    nframes = 1
    comptype = 1
    compname = 1

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

    def getTrainingSet(self, readFile, inputDim, totalFrames):
        self.setParams(readFile)
        # prepare array of right shape
        sampwidth = readFile.getsampwidth()
        numberOfSets = (totalFrames*sampwidth) / inputDim

        if numberOfSets % 1 > 0:
            numberOfSets = int(numberOfSets) + 1
        else:
            numberOfSets = int(numberOfSets)

        x_train = np.empty((numberOfSets, inputDim), dtype=np.ubyte)

        if inputDim/sampwidth % 1 > 0:
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

    def setParams(self, inputFile):
        self.nchannels = inputFile.getnchannels()
        self.sampwidth = inputFile.getsampwidth()
        self.framerate = inputFile.getframerate()
        self.nframes = inputFile.getnframes()
        self.comptype = inputFile.getcomptype()
        self.compname = inputFile.getcompname()

    def writeCover(self, inputTrack, encoder):
        newCover = self.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\generated" + os.sep + "newCover" + random.randint(0, 100).__str__() +".wav", 'wb')
        newCover.setnchannels(self.nchannels)
        newCover.setsampwidth(self.sampwidth)
        newCover.setframerate(self.framerate)
        newCover.setcomptype(self.comptype, self.compname)

        res = encoder.predict(inputTrack)
        intArray = []
        for entry in res:
            entry = self.deNormilize(entry)
            list = entry.tolist()
            for i in range(0, len(list)):
                intArray.append(list[i])

        newCover.writeframes(bytes(intArray))
        print(newCover.getparams())
        newCover.close()
        return newCover


    def normalize(self, inputSet):
        return inputSet.astype('float32') / 255.


    def deNormilize(self, outputSet):
        return (outputSet * 255).astype('uint8')

    def getAmplitudes(self, readFile):
        totalFrames = readFile.getnframes()
        currentPos = 0
        max = -10000000
        min = 10000000


        while currentPos < totalFrames:

            frame = readFile.readframes(1)
            int_value = int.from_bytes(frame, byteorder='little')

            if int_value > max:
                max = int_value
            if int_value < min:
                min = int_value

            currentPos += 1

        maxUint = np.uint32(max)
        minUint = np.uint32(min)
        print("Max amplitude: ", maxUint.astype('int32'))
        print("Min amplitude: ", minUint.astype('int32'))