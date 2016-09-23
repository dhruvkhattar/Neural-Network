#! /usr/bin/env python
#######################################
#     Author : Dhruv Khattar         #
#####################################

from __future__ import division
import numpy as np
import scipy.misc
import pdb

DIGITS = [1, 3, 6]
START = 21
SIZE = 32
LAYERS = 3
LSIZE = [0, 65, 8, 2]
ETA = 2

def f(x):
    '''
    Activation Function
    '''

    return 1 / (1 + np.exp(-x))

def df(x):
    '''
    Derivative of Activation function
    '''

    return scipy.misc.derivative(f, x)


class DataHandler():
    
    def __init__(self):
        
        self.arrays = []
        self.digitizedArrays = []
        self.T = []
        self.tT = [] #test
        self.tarrays = [] #test


    def convertToFloat(self, x):

        return float(x)


    def preprocess(self):
        with open('data/optdigits-orig.tra') as f:
            lines = f.readlines()
            idx = START
            while idx < len(lines):
                if int(lines[idx + SIZE].strip()) not in DIGITS:
                    idx += SIZE + 1
                    continue
                temp = np.zeros(LSIZE[-1])
                no = int(lines[idx+SIZE].strip())
                if no == DIGITS[0]:
                    temp[0] = 0
                    temp[1] = 0
                elif no == DIGITS[1]:
                    temp[0] = 0
                    temp[1] = 1
                else:
                    temp[0] = 1
                    temp[1] = 0
                self.T.append(temp)
                ct = 0
                arr = []
                while ct < SIZE:
                    arr.append(map(self.convertToFloat, list(lines[idx+ct].strip())))
                    ct += 1

                resizedArr = scipy.misc.imresize(np.asarray(arr), (8, 8), 'nearest') / 255
                self.digitizedArrays.append(resizedArr)
                self.arrays.append(resizedArr.flatten())
                idx += SIZE + 1
            print 'Training Data:', len(self.arrays)

        #test
        with open('data/optdigits-orig.cv') as f:
            lines = f.readlines()
            idx = START
            while idx < len(lines):
                if float(lines[idx + SIZE].strip()) not in DIGITS:
                    idx += SIZE + 1
                    continue
                temp = np.zeros(LSIZE[-1])
                no = int(lines[idx+SIZE].strip())
                if no == DIGITS[0]:
                    temp[0] = 0
                    temp[1] = 0
                elif no == DIGITS[1]:
                    temp[0] = 0
                    temp[1] = 1
                else:
                    temp[0] = 1
                    temp[1] = 0
                self.tT.append(temp)
                ct = 0
                arr = []
                while ct < SIZE:
                    arr.append(map(self.convertToFloat, list(lines[idx+ct].strip())))
                    ct += 1
                resizedArr = scipy.misc.imresize(np.asarray(arr), (8, 8), 'nearest') / 255
                self.tarrays.append(resizedArr.flatten())
                idx += SIZE + 1
            print 'Test Data:', len(self.tarrays)

class NeuralNet():

    def __init__(self, data):
        
        self.data = data.arrays
        self.tdata = data.tarrays
        self.tT = data.tT
        self.T = data.T
        self.layers = []
        for i in xrange(LAYERS):
            self.layers.append(Layer(LSIZE[i], LSIZE[i+1]))
        self.layers[0].unit[0] = 1

    
    def run(self):

        ct = 0
        while ct < 100:
            test = 0
            for it in range(len(self.data)):
                self.layers[0].unit[1:] = self.data[it]
                net.forwardPropagation()
                test +=  np.dot(self.T[it] - self.layers[LAYERS-1].unit, self.T[it] - self.layers[LAYERS-1].unit)
                net.backPropagation(it)
            print 'Iteration: ', ct
            print 'Training Error: ', test/2
            self.test()
            ct += 1
        np.save('hw', [self.layers[1].w, self.layers[2].w])
        np.savetext('hw1', self.layers[1].w)
        np.savetext('hw2', self.layers[2].w)

    
    def test(self):
        
        test = 0
        ans = 0
        for it in range(len(self.tdata)):
            self.layers[0].unit[1:] = self.tdata[it]
            for i in range(1, LAYERS):
                self.layers[i].net = np.dot(self.layers[i].w, self.layers[i-1].unit)
                self.layers[i].unit = np.asarray(map( f, self.layers[i].net))
            dist1 = self.layers[LAYERS-1].unit - [0, 0]
            dist1 = np.dot(dist1, dist1)
            dist2 = self.layers[LAYERS-1].unit - [0, 1]
            dist2 = np.dot(dist2, dist2)
            dist3 = self.layers[LAYERS-1].unit - [1, 0]
            dist3 = np.dot(dist3, dist3)
            mindist = min(dist1, dist2, dist3)
            if self.tT[it][0] == 0 and self.tT[it][1] == 0 and mindist != dist1:
                ans += 1
            if self.tT[it][0] == 0 and self.tT[it][1] == 1 and mindist != dist2:
                ans += 1
            if self.tT[it][0] == 1 and self.tT[it][1] == 0 and mindist != dist3:
                ans += 1
            test +=  np.dot(self.tT[it] - self.layers[LAYERS-1].unit, self.tT[it] - self.layers[LAYERS-1].unit)
        print 'Misclassified Points: ', ans
        print 'Test Error', test/2



    def forwardPropagation(self):

        for i in range(1, LAYERS):
            self.layers[i].net = np.dot(self.layers[i].w, self.layers[i-1].unit)
            self.layers[i].unit = np.asarray(map(f, self.layers[i].net))


    def backPropagation(self, no):

        mult = self.T[no] - self.layers[LAYERS-1].unit
        for i in range(LAYERS-1, 0, -1):
            w = np.zeros_like(self.layers[i].w)
            self.layers[i].delta = np.multiply(mult, map(df, self.layers[i].net))
            for j in xrange(self.layers[i].size):
                w[j] = ETA * self.layers[i].delta[j] * self.layers[i-1].unit
            self.layers[i].w += w
            mult = []
            for j in xrange(self.layers[i-1].size):
                mult.append(np.dot(self.layers[i].delta, self.layers[i].w[:, j]))


class Layer():

    def __init__(self, prevSize, curSize):

        self.size = curSize
        self.net = np.zeros(self.size)
        self.unit = np.zeros(self.size)
        self.w = np.random.rand(curSize, prevSize)
        self.delta = np.zeros(self.size)


if __name__ == '__main__':
    data = DataHandler()
    data.preprocess()
    net = NeuralNet(data)
    net.run()
