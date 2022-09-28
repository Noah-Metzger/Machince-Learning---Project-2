import numpy as np
import pandas as pd
import math
import copy
from evaluation import *

class KNN:
    def __init__(self, df, test, test_y, train, train_y, k, truth):
        self.df = df
        self.test = test
        self.test_y = test_y
        self.train = train
        self.train_y = train_y
        self.k = k
        self.truthIndex = truth

    def lpNorm(self, x, y, p):
        """

        :param x:
        :param y:
        :param p:
        :return:
        """

        sums = 0
        for i in range(len(x)):
            sums += pow(x[i] - y[i], p)
        return pow(sums, (1 / p))

    def kernel(self, x, y, sigma):
        """

        :param x:
        :param y:
        :param sigma:
        :return:
        """
        r = 1 / (2 * sigma)
        return math.exp(-r * self.lpNorm(x, y, 2))

    def predictInstance(self, testInstance, testTruth, trainingSet, trainingTruth, isClassification, bandwidth, isEditedKNN, error):
        neighbors = []
        classes = []
        indices = []
        for i, row in trainingSet.iterrows():
            dist = self.lpNorm(np.array(row), np.array(testInstance), 2)
            neighbors.append(dist)
            classes.append(trainingTruth[i])
            indices.append(i)

        for i in range(len(neighbors)):
            for j in range(0, len(neighbors) - i - 1):
                if neighbors[j] > neighbors[j + 1]:
                    neighbors[j], neighbors[j + 1] = neighbors[j + 1], neighbors[j]
                    classes[j], classes[j + 1] = classes[j + 1], classes[j]
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]

        nearestNeighbors = classes[:self.k]
        nearestIndices = indices[:self.k]

        if isClassification:
            vote = np.unique(nearestNeighbors)
            count = []
            for i in range(len(vote)):
                count.append(0)

            for i in nearestNeighbors:
                for j, cl in enumerate(vote):
                    if i == cl:
                        count[j] += 1
            predictedClass = vote[np.argmax(count)]
            if isEditedKNN:
                isSameClass = False
                if predictedClass == testTruth:
                    isSameClass = True
                return []
            return predictedClass
        else:
            nom = 0
            dom = 0
            for i in nearestIndices:
                kern = self.kernel(trainingSet.loc[i], testInstance, bandwidth)
                nom += kern * trainingTruth[i]
                dom += kern
            predictedValue = (nom / dom)
            if isEditedKNN:
                isWithinError = False
                if abs(predictedValue-testTruth) < error:
                    isWithinError = True
                return [predictedValue, isWithinError]

            return predictedValue

    def knnRegular(self, isClassification, bandwidth):
        truth = []
        predictions = []
        for i, testRow in self.test.iterrows():
            predictions.append(self.predictInstance(testRow, self.test_y[i], self.train, self.train_y,isClassification, bandwidth, False, 0))
            truth.append(self.test_y[i])
        return [predictions, truth]

    def knnEdited(self, isClassification, bandwidth, error):
        df = copy.copy(self.df)
        isPerformanceIncreasing = True
        prevProformance = -1
        while isPerformanceIncreasing:
            isPerformanceIncreasing = False
            truth = []
            predicted = []
            for i, test_row in df.iterrows():
                rest = df.drop(i, axis=0)
                train_y = rest[rest.columns[self.truthIndex]]
                train = rest.drop(rest.columns[self.truthIndex], axis=1)
                test_y = test_row[self.truthIndex]
                test = np.array(list(test_row)[:self.truthIndex])
                predictedResponse = self.predictInstance(test, test_y, train, train_y, isClassification, bandwidth, True, error)
                if not predictedResponse[1]:
                    df = df.drop([i])
                    isPerformanceIncreasing = True

                if isClassification:
                    truth.append(test_y)
                    predicted.append(predictedResponse[0])
                else:
                    truth.append(True)
                    predicted.append(predictedResponse[1])

            if not isClassification:
                whole = [False, True]
                e = Evaluation(predicted, truth, whole)
            else:
                e = Evaluation(predicted, truth, self.df[self.df.columns[self.truthIndex]])

            if isClassification:
                newPreformance = e.precision() + e.recall()
            else:
                newPreformance = sum(e.precision()) + sum(e.recall())

            if newPreformance < prevProformance:
                isPerformanceIncreasing = True
            prevProformance = newPreformance
        return df
