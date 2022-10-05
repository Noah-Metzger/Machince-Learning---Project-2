import numpy as np
import pandas as pd
import math
import copy
from evaluation import *

class KNN:
    def __init__(self, df, test, test_y, train, train_y, truth):
        """
        Constructor for KNN class

        :param df: The whole dataframe of the dataset
        :param test: The test set of the dataframe
        :param test_y: The test's class set of the dataframe
        :param train: The train set of the dataframe
        :param train_y: The train's class set of the dataframe
        :param k: The k number of neighbors
        :param truth: The index of the dataframe's class feature
        """
        self.df = df
        self.test = test
        self.test_y = test_y
        self.train = train
        self.train_y = train_y
        self.truthIndex = truth

    def lpNorm(self, x, y, p):
        """
        Helper method that calculates the L_p norm of two points

        :param x: A numpy array, Python list or Pandas Series containing one of two point to have there distance measured.
        :param y: A numpy array, Python list or Pandas Series containing one of two point to have there distance measured.
        :param p: p from the L_p norm function
        :return: L_p norm distance between two points
        """

        sums = 0
        for i in range(len(x)):
            sums += pow(float(x[i]) - float(y[i]), p)
        return pow(sums, (1 / p))

    def RBF(self, x, y, sigma):
        """
        Helper method that calculates the Radial Basis Function (RBF)

        :param x: A numpy array, Python list or Pandas Series containing one of two point to have there kernel distance measured.
        :param y: A numpy array, Python list or Pandas Series containing one of two point to have there kernel distance measured.
        :param sigma: Bandwidth of the kernel function
        :return: RBF distance
        """
        r = 1 / (2 * sigma)
        out = math.exp(-r * self.lpNorm(x, y, 2))
#         print(out)
        return out

    def predictInstance(self, k, testInstance, testTruth, trainingSet, trainingTruth, isClassification, bandwidth, isEditedKNN, error):
        """
        Predicts the class category or values of a single instance of a dataset

        :param testInstance: The instance whose class is being predicted
        :param testTruth: The class of the instance whose class is being predicted
        :param trainingSet: The training set
        :param trainingTruth: The train set truth class
        :param isClassification: Boolean for whether solving for classification or regression
        :param bandwidth: Bandwidth used in the kernel (Regression only)
        :param isEditedKNN: Boolean for whether method is being used for edited-KNN or not
        :param error: Maximum error that a regression prediction can have to be considered correct
        :return: The predicted class or value, if being used by edited-KNN
        """
        neighbors = []
        classes = []
        indices = []
        #Iterate through all the instances in the training set and record there distance from the test instance, class or value, and index in the dataframe
        for i, row in trainingSet.iterrows():
            dist = self.lpNorm(np.array(row), np.array(testInstance), 2)
            neighbors.append(dist)
            classes.append(trainingTruth[i])
            indices.append(i)

        #Sort all distance, class/value, and index array by distance from the test instance
        for i in range(len(neighbors)):
            for j in range(0, len(neighbors) - i - 1):
                if neighbors[j] > neighbors[j + 1]:
                    neighbors[j], neighbors[j + 1] = neighbors[j + 1], neighbors[j]
                    classes[j], classes[j + 1] = classes[j + 1], classes[j]
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]

        #Reduce the distance and index array to only contain the k-nearest neighbors
        nearestNeighbors = classes[:k]
        nearestIndices = indices[:k]

        #If is classification problem
        if isClassification:
            #Count the classes of the k-nearest-neighbors
            vote = np.unique(nearestNeighbors)
            count = []
            for i in range(len(vote)):
                count.append(0)

            for i in nearestNeighbors:
                for j, cl in enumerate(vote):
                    if i == cl:
                        count[j] += 1
            #Predicted class is the class of the most k-nearest-neighbors
            predictedClass = vote[np.argmax(count)]
            #If edited KNN problem
            if isEditedKNN:
                #If predicted class is same as truth class, return True otherwise return false
                isSameClass = False
                if predictedClass == testTruth:
                    isSameClass = True
                return [predictedClass, isSameClass]
            return predictedClass
        # If is regression problem
        else:
            nom = 0
            dom = 0
            #Apply the Gaussian kernel smoother to all predicted values
#             print(trainingSet)
            for i in nearestIndices:
                kern = self.RBF(np.array(trainingSet.loc[i]), np.array(testInstance), bandwidth)
                nom += kern * trainingTruth[i]
                dom += kern
            predictedValue = (nom / dom)
            if dom == 0:
                predictedValue = 0
            #If edited KNN problem
            if isEditedKNN:
                #if predicted value
                isWithinError = False
                if abs(predictedValue-testTruth) < error:
                    isWithinError = True
                return [predictedValue, isWithinError]

            return predictedValue

    def knnRegular(self, k, isClassification, bandwidth):
        """
        Normal KNN algoritm for both classification and regression

        :param isClassification: Whether problem is classification or regression.
        :param bandwidth: Bandwidth for the kernel function
        :return: Two arrays, first the predicted values or classes and the second truth values or classes
        """
        truth = []
        predictions = []
        #Iterate through each instance in a test set and run KNN
        for i, testRow in self.test.iterrows():
            predictions.append(self.predictInstance(k, testRow, self.test_y[i], self.train, self.train_y,isClassification, bandwidth, False, 0))
            truth.append(self.test_y[i])
        return [predictions, truth]

    def knnEdited(self, k, isClassification, bandwidth, error):
        """
        Edited KNN algorithm

        :param isClassification: Whether problem is classification or regression.
        :param bandwidth: Bandwidth for the kernel function
        :param error: If regression, maximum error from truth class to be considered correct.
        :return: A edited-KNN dataframe
        """
        df = copy.copy(self.df)
        isPerformanceIncreasing = True
        prevProformance = -1
        #Loop until Preformance stays the same or no instances are removed
        while isPerformanceIncreasing:
            isPerformanceIncreasing = False
            truth = []
            predicted = []
            #Iterate through each row in dataset and run KNN
            for i, test_row in df.iterrows():
                rest = df.drop(i, axis=0)
                train_y = rest[rest.columns[self.truthIndex]]
                train = rest.drop(rest.columns[self.truthIndex], axis=1)
                test_y = test_row[self.truthIndex]
                test = np.array(list(test_row)[:self.truthIndex])
                predictedResponse = self.predictInstance(k, test, test_y, train, train_y, isClassification, bandwidth, True, error)
                print(predictedResponse, isClassification)
                #If predicted class/value incorrect then remove instance from the dataset, and continue the loop to the next dataset
                if not predictedResponse[1]:
                    df = df.drop([i])
                    isPerformanceIncreasing = True

                #To evaluate preformance of algorithm to determine if it is time to stop loop
                if isClassification:
                    truth.append(test_y)
                    predicted.append(predictedResponse[0])
                else:
                    truth.append(True)
                    predicted.append(predictedResponse[1])

            #Calculates preformance of predictions
            if not isClassification:
                whole = [False, True]
                e = Evaluation(predicted, truth, whole)
                newPreformance = sum(e.precision()) + sum(e.recall())
            else:
                e = Evaluation(predicted, truth, self.df[self.df.columns[self.truthIndex]])
                newPreformance = sum(e.precision()) + sum(e.recall())

            #if preformance increases, continue to next iteration
            if newPreformance > prevProformance:
                isPerformanceIncreasing = True
            prevProformance = newPreformance
        return df

    def Kmeans(self, k, maxIter):
        """
        K-Means algorithm

        :param k: Number of clusters
        :param maxIter: Maximum number of iterations
        :return: Array of classes
        """

        #selects k existing points to be the initial cluster centers
        kPoints = self.df.sample(k)
        centers = []
        for i, r in kPoints.iterrows():
            centers.append(np.array(r))

        finalClusters = []
        for z in range(maxIter):
            binn = []
            for i in range(k):
                binn.append([])
            #Iterates through each instance, decides which cluster is closer and puts the instance into its corresponding bin for each cluster
            for i, row in self.df.iterrows():
                dist = []
                for center in centers:
                    dist.append(lpNorm(row, center, 2))
                binn[np.argmin(np.array(dist))].append(i)
            finalClusters = binn
            #Finds the average point for each cluster and assigns the average point as the new cluster centers
            for index, cluster in enumerate(binn):
                avgInst = np.empty([1, 1])
                for i, inst in enumerate(cluster):
                    row = np.array(self.df.loc[inst])
                    if i == 0:
                        avgInst = row
                    else:
                        for j, val in enumerate(row):
                            avgInst[j] += val
                for i in range(len(avgInst)):
                    avgInst[i] /= len(cluster)
                centers[index] = avgInst

        #once loop complete, return array of cluster centeriods.
        output = []
        for i, cluster in enumerate(finalClusters):
            temp = pd.DataFrame()
            for j in cluster:
                temp = temp.append(df.loc[j])
            output.append(temp)

        return output