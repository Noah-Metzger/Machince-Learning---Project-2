import numpy as np
import pandas as pd
import math

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
        print(x)
        print(y)
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
        return math.exp(-r * lpNorm(x, y, 2))

    def predictInstance(self, testInstance, testTruth, trainingSet, isClassification, bandwidth):
        neighbors = []
        classes = []
        indices = []
        for i, row in self.train.iterrows():
            dist = self.lpNorm(np.array(row), np.array(testInstance), 2)
            neighbors.append(dist)
            classes.append(self.train_y[i])
            indices.append(i)

        for i in range(len(neighbors)):
            for j in range(0, len(neighbors) - i - 1):
                if neighbors[j] > neighbors[j + 1]:
                    neighbors[j], neighbors[j + 1] = neighbors[j + 1], neighbors[j]
                    classes[j], classes[j + 1] = classes[j + 1], classes[j]
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]

        nearestNeighbors = classes[:k]
        nearestIndices = indices[:k]

        if isClassification:
            vote = np.unique(nearestNeighbors)
            count = []
            for i in range(len(vote)):
                count.append(0)

            for i in nearestNeighbors:
                for j, cl in enumerate(vote):
                    if i == cl:
                        count[j] += 1
            return vote[np.argmax(count)]
        else:
            nom = 0
            dom = 0
            for i in nearestIndices:
                kern = self.kernel(train.loc[i], testInstance, bandwidth)
                nom += kern * train_y[i]
                dom += kern

            return (nom / dom)

    def knnRegular(self, isClassification, bandwidth):
        truth = []
        predictions = []
        for i, testRow in self.test.iterrows():
            predictions.append(self.predictInstance(testRow, self.test_y[i], self.train, isClassification, bandwidth))
            truth.append(self.test_y[i])
        return [predictions, truth]

    def kknEdited(self, isClassification, bandwidth, error):
        deletedIndices = []
        df = copy.copy(self.df)
        for i, test in df.iterrows:
            rest = df.drop(i, axis=0)
            train_y = rest.iloc[self.truth]
            train = rest.drop([self.truth], axis=1)
            test_y = row[self.truth]
            test = np.array(list(row)[:self.truth])
            predictInstance(test, test_y, train)
