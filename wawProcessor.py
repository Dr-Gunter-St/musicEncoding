import os
import random
import wave
import numpy as np
import matplotlib.pyplot as plt


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

    def getTrainingSet(self, readFile, inputDim, normalize=True, normalization_factor=255.):
        self.setParams(readFile)
        # prepare array of right shape
        totalFrames = readFile.getnframes()
        numberOfSets = (totalFrames) / inputDim

        # Omg, this is a round up part
        if numberOfSets % 1 > 0:
            numberOfSets = int(numberOfSets) + 1
        else:
            numberOfSets = int(numberOfSets)

        detype = np.ubyte
        if normalization_factor == 65535.:
            detype = np.uint16

        x_train = np.empty((numberOfSets, inputDim), dtype=detype)

        # miscellaneous vars
        currentPos = 0
        currentSet = 0
        readFile.setpos(currentPos)

        while currentPos + inputDim < totalFrames:

            currentSample = 0
            int_set = []
            # proper preprocessing
            while currentSample < inputDim:
                frame = readFile.readframes(1)
                int_value = int.from_bytes(frame, byteorder='little')
                int_set.append(int_value)
                currentSample += 1

            x_train[currentSet] = np.array(int_set)
            currentPos += inputDim
            currentSet += 1

        if currentPos < totalFrames:

            currentSample = 0
            int_set = []

            while currentSample < totalFrames - currentPos:
                frame = readFile.readframes(1)
                int_value = int.from_bytes(frame, byteorder='little')
                int_set.append(int_value)
                currentSample += 1

            while len(int_set) < inputDim:
                int_set.append(0)
            x_train[currentSet] = np.array(int_set)

        if normalize:
            return self.normalize(x_train, normalization_factor)
        return x_train

    def setParams(self, inputFile):
        self.nchannels = inputFile.getnchannels()
        self.sampwidth = inputFile.getsampwidth()
        self.framerate = inputFile.getframerate()
        self.nframes = inputFile.getnframes()
        self.comptype = inputFile.getcomptype()
        self.compname = inputFile.getcompname()

    def writeCover(self, inputTrack, encoder, de_normalize=True, de_normalization_factor=255):
        newCover = self.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\generated" + os.sep + "newCover" + random.randint(0, 100).__str__() +".wav", 'wb')
        newCover.setnchannels(self.nchannels)
        newCover.setsampwidth(self.sampwidth)
        newCover.setframerate(self.framerate)
        newCover.setcomptype(self.comptype, self.compname)
        self.printparams()

        res = encoder.predict(inputTrack)
        typefunc = np.uint8
        if de_normalization_factor == 65535:
            typefunc = np.uint16
        for entry in res:
            if de_normalize:
                entry = self.deNormilize(entry, de_normalization_factor)
            for i in range(0, len(entry)):
                newCover.writeframes(int.to_bytes(typefunc(entry[i]).item(), self.sampwidth, 'little'))

        # int.to_bytes(np.uint16(entry[0]).item(),2,'little')
        print(newCover.getparams())
        newCover.close()
        return newCover


    def normalize(self, inputSet, normalization_factor=255.):
        return inputSet.astype('float32') / normalization_factor


    def deNormilize(self, outputSet, de_normalization_factor=255):
        stype = 'uint8'
        if de_normalization_factor == 65535:
            stype = 'uint16'
        return (outputSet * de_normalization_factor).astype(stype)

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

        print("Max amplitude: ", max)
        print("Min amplitude: ", min)
        return max, min

    def printparams(self):
        print("nchannels: ", self.nchannels, ", sampwidth: ", self.sampwidth,
                    ", framerate: ", self.framerate, ", nframes: ", self.nframes,
                    ", comptype: ", self.compname, ", framerate: ", self.compname)

    def drawDensity(self, inputSet, amplitude, label):
        arrayedFile = self.getTrainingSet(inputSet, inputSet.getnframes(), normalize=False, normalization_factor=float(amplitude))
        plt.hist(arrayedFile[0])
        plt.xlabel('Value', fontsize=15)
        plt.ylabel('Frequency', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(label, fontsize=15)
        plt.show()

    def getRNNTrainSequences(self, source, result, seq_size, normalize=True, normalization_factor=255.):
        wholeSourceSequence = self.getTrainingSet(source, source.getnframes(), normalize, normalization_factor)[0]
        wholeResultingSequence = self.getTrainingSet(result, result.getnframes(), normalize, normalization_factor)[0]

        if (len(wholeSourceSequence) > len(wholeResultingSequence)):
            wholeSourceSequence = wholeSourceSequence[:len(wholeResultingSequence)-1]

        return self.to_sequences(seq_size, wholeSourceSequence, wholeResultingSequence)

    def to_sequences(self, seq_size, obs1, obs2):
        x = []
        y = []

        for i in range(len(obs1) - seq_size):
            # print(i, "/", len(obs1))
            window = obs1[i:(i + seq_size)]
            after_window = obs2[i + seq_size]
            window = [[x] for x in window]
            # print("{} - {}".format(window,after_window))
            x.append(window)
            y.append(after_window)

        return np.array(x), np.array(y)

    def getValidationSet(self, x_train, y_train):
        size = len(x_train)
        x_test = x_train[int(size*0.8):]
        y_test = y_train[int(size*0.8):]
        x_train = x_train[:int(size*0.8)]
        y_train = y_train[:int(size*0.8)]

        return x_train, y_train, x_test, y_test

    def writeRNNCover(self, inputSequences, model):
        newCover = self.openWave("C:\\Users\\NKF786\\PycharmProjects\\musicEncoding\\generated" + os.sep + "newCoverRNN" + random.randint(0, 100).__str__() +".wav", 'wb')
        newCover.setnchannels(self.nchannels)
        newCover.setsampwidth(self.sampwidth)
        newCover.setframerate(self.framerate)
        newCover.setcomptype(self.comptype, self.compname)
        self.printparams()

        res = model.predict(inputSequences)
        typefunc = np.uint8
        de_normalization_factor = 255
        if self.sampwidth == 2:
            typefunc = np.uint16
            de_normalization_factor = 65535
        for entry in res:
            entry = self.deNormilize(entry, de_normalization_factor)
            for i in range(0, len(entry)):
                newCover.writeframes(int.to_bytes(typefunc(entry[i]).item(), self.sampwidth, 'little'))

        # int.to_bytes(np.uint16(entry[0]).item(),2,'little')
        print(newCover.getparams())
        newCover.close()
        return newCover