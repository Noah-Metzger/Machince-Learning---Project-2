import random
import sys

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

class Preprocessor:
    def __init__(self, df, truth, name):
        """
        Constructor for Preprocessor class.  All preprocessing logic is applied to the preprocessor object.

        :param df: data table
        :param truth: index of the ground truth column
        :param name: String name of the dataset
        """
        self.df = df
        self.dfName = name
        self.truthCol = df.iloc[:, truth]
        self.truthColIndex = truth
        self.checkItems = ["?"]

    def removesmissingvalues(self):
        """
        Removes each observation that contains a missing value.
        """
        #Array of row indices that contain missing values
        missingRows = []
        #Checking by row for missing values
        for i, row in self.df.iterrows():
            isMissing = False
            for j, value in row.items():
                for k in self.checkItems:
                    if value == k or value == math.nan:
                        isMissing = True
            if isMissing:
                missingRows.append(i)
        #Removal of rows/instances with missing values
        self.df.drop(missingRows, axis=0, inplace=True)

    def fillmean(self):
        """
        Fills each missing value with the mean value of the missing value's attribute.
        """
        #Checks if column contains a missing value
        for col in self.df:
            missingIndex = []
            for index, value in self.df[col].items():
                for k in self.checkItems:
                    if value == k or value == math.nan:
                        isMissing = True
            #If column contains missing value, take mean and replace missing values with mean
            if len(missingIndex) > 0:
                newCol = self.df[col]
                newCol.drop(missingIndex)
                mean = newCol.mean()
                for i in missingIndex:
                    self.df[col][i] = mean

    def fillforward(self):
        """
        Forward fills all missing values.
        """
        #Checks for missing values
        for col in self.df:
            for index, value in self.df[col].items():
                for k in self.checkItems:
                    #If missing value found, replace with value below the missing value
                    if value == k or value == math.nan:
                        if index + 1 >= len(self.df[col]):
                            self.df[col][index] = self.df[col][0]
                        else:
                            self.df[col][index] = self.df[col][index + 1]

    def fillbackward(self):
        """
        Backward fills all missing values.
        """
        # Checks for missing values
        for col in self.df:
            for index, value in self.df[col].items():
                for k in self.checkItems:
                    # If missing value found, replace with value below the missing value
                    if value == k or value == math.nan:
                        if index - 1 < 0:
                            self.df[col][index] = self.df[col][len(self.df[col]) - 1]
                        else:
                            self.df[col][index] = self.df[col][index - 1]

    def labelencode(self):
        """
        Label encodes all categorical attributes.
        """
        #Makes a list of all unique the values in each column
        for col in self.df:
            if type(self.df[col][0]) == str:
                labels = []
                for index, value in self.df[col].items():
                    isDup = False
                    for i in labels:
                        if i == value:
                            isDup = True
                    if not isDup:
                        labels.append(value)
                #Replaces the value with its index of the value in the unique values array
                for index, value in self.df[col].items():
                    for i in labels:
                        if i == value:
                            self.df[col][index] = labels.index(value)

    def onehotencoding(self):
        """
        One hot encodes all categorical attributes
        """
        # Makes a list of all unique the values in each column
        for col in self.df:
            if type(self.df[col][0]) == str:
                labels = []
                for index, value in self.df[col].items():
                    isDup = False
                    for i in labels:
                        if i == value:
                            isDup = True
                    if not isDup:
                        labels.append(value)
                #Inserts a column for each unique value and inserts a 1 for each occurrance of that value and all else 0's
                for i in labels:
                    temp = np.zeros(self.df[col].size)
                    for index, value in self.df[col].items():
                        if i == value:
                            temp[index] = 1
                    self.df.insert(col, i, temp)
                self.df.drop(self.df.columns[[col + len(labels)]], axis=1, inplace=True)

    def binning(self, columns, BIN_NUMBER):
        """
        Takes a numerical attribute and converts it into a categorical attribute by taking a specified number of desired categories and splitting them into a "bin" for each category.
        The separating into bins are based on a sorted list of the values and the list is split equally by order.  Each split section is put into a separate bin.  Feature is label encoded

        :param columns: The desired numerical attribute
        :param BIN_NUMBER: The desired number of "bins" or categories
        """

        for col in columns:
            # Bubble sorts each col, when values are swapped in col array, the same indices are swapped indices array to keep track of values position.
            tempCol = self.df.iloc[:, col]
            indices = np.array(list(range(0,len(tempCol))))
            for i in range(len(tempCol)):
                for j in range(0, len(tempCol) - i - 1):
                    if tempCol[i] > tempCol[j]:
                        temp = tempCol[i]
                        tempCol[i] = tempCol[j]
                        tempCol[j] = temp

                        tempIndex = indices[i]
                        indices[i] = indices[j]
                        indices[j] = tempIndex

            #The sorted array of values split into equal sections, each section is a bin
            bins = np.array_split(tempCol, BIN_NUMBER)
            binIndices = np.array_split(indices, BIN_NUMBER)

            #The new categorical value is the bin's index that the value is contained within
            for i, bin in enumerate(binIndices):
                for ind in bin:
                    self.df.iloc[:, col][ind] = i

    def deleteFeature(self, index):
        """
        Deletes a column of the dataset based on the columns index

        :param index: The index of the column to be deleted
        """
        self.df = self.df.drop([index],axis=1)
        #Changes the index of the ground truth feature as it's position changes
        if self.truthColIndex > index:
            self.truthColIndex -= 1


    def shuffle(self):
        """
        Takes 10% or at least one features if less than 10 features and shuffles values in that feature.  Meant to introduce noise into the dataset.
        """

        #Finds number of features to shuffle
        randoms = int(self.df.shape[1] * 0.1)
        if randoms == 0:
            randoms = 1

        #Sets seed for random number generators to have reproducable results
        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        #Selects a random feature and shuffles the values of the feature.
        for i in range(randoms):
            r = random.randint(0, self.df.shape[1] - 1)
            if r == self.truthColIndex:
                while r != self.truthColIndex:
                    r = random.randint(0, self.df.shape[1] - 1)
            feature = np.array(self.df.iloc[:,r])
            np.random.shuffle(feature)
            self.df.iloc[:, r] = feature