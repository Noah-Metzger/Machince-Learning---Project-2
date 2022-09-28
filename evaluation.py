from preprocess import *
from numpy import random
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

class Evaluation:
    def __init__(self, pred, truth, whole):
        """
        Constructor for Evaluation class.  All classification evaluation logic is applied to the evaluation object.

        :param df: data table
        :param pred: Predicted classification
        :param truth: Ground truth classification
        """

        self.pred = pred
        self.truth = truth
        self.label = self.getlabels(whole)

    def getlabels(self, col):
        """
        Returns a non-duplicate list of the possible categorical values of a attribute.

        :param col: Categorical attribute
        :type col: Pandas Series
        :return: list of possible categories
        :rtype: list
        """

        return list(np.unique(col))

    def printConfusionMatrix(self):
        matrix = self.getconfusionmatrix()
        print("***Confusion Matrix***")
        for i in matrix:
            line = ""
            for j in i:
                line += str(j) + " "
            print(line)

    def getconfusionmatrix(self):
        """
        Returns a confusion matrix for the results of a classifier

        :return: a confusion matrix as a 2d list.
        """
        matrix = []
        for i in range(len(self.label)):
            matrix.append([0] * len(self.label))
        for i in range(len(self.pred)):
            matrix[self.label.index(self.pred[i])][self.label.index(self.truth[i])] += 1
        return matrix

    def truepositive(self, conmat):
        """
        Returns the count of true positives from a classification

        :param conmat: Confusion matrix
        :return: integer count of true positives
        """
        TP = []
        for i in range(len(conmat)):
            TP.append(conmat[i][i])
        return TP

    def truenegative(self, conmat):
        """
        Returns the count of true negative from a classification.

        :param conmat: Confusion matrix.
        :return: integer count of true negative.
        """
        TN = []
        for i in range(len(conmat)):
            count = 0
            for j in range(len(conmat)):
                for k in range(len(conmat)):
                    if j != i and k != i:
                        count += conmat[j][k]
            TN.append(count)
        return TN

    def falsepositive(self, conmat):
        """
        Returns the count of false positives from a classification.

        :param conmat: Confusion matrix.
        :return: integer count of false positives.
        """
        FP = []
        for i in range(len(conmat)):
            count = 0
            for j in range(len(conmat)):
                if i != j:
                    count += conmat[i][j]
            FP.append(count)
        return FP

    def falsenegative(self, conmat):
        """
        Returns the count of false negative from a classification.

        :param conmat: Confusion matrix.
        :return: integer count of false negative.
        """
        FN = []
        for i in range(len(conmat)):
            count = 0
            for j in range(len(conmat)):
                if i != j:
                    count += conmat[j][i]
            FN.append(count)
        return FN

    def precision(self):
        """
        Returns the precision values of classifier

        :return: a list of the precision of each category.
        """
        conmat = self.getconfusionmatrix()
        TP = self.truepositive(conmat)
        FP = self.falsepositive(conmat)
        prec = []
        for i in range(len(conmat)):
            if (TP[i] + FP[i]) == 0:
                prec.append(0)
            else:
                prec.append((TP[i]) / (TP[i] + FP[i]))

        # self.printResults("precision", prec)
        return prec

    def recall(self):
        """
        Returns the recall values of a classifier

        :param pred: Predicted classification from the classifier
        :param truth: Ground truth classificati on
        :param label: List of possible categories.
        :return: a list of the recall of each category.
        """
        conmat = self.getconfusionmatrix()
        TP = self.truepositive(conmat)
        FN = self.falsenegative(conmat)
        rec = []
        for i in range(len(conmat)):
            if (TP[i] + FN[i]) == 0:
                rec.append(0)
            else:
                rec.append((TP[i]) / (TP[i] + FN[i]))

        # self.printResults("recall", rec)
        return rec

    def fscore(self, b):
        """
        Returns the fScore values of a classifier.

        :param b: The importance of recall over precision.
        :param pred: Predicted classification from the classifier.
        :param truth: Ground truth classification
        :param label: List of possible categories.
        :return: a list of the fscores of each category.
        """
        prec = self.precision()
        rec = self.recall()
        f = []
        for i in range(len(self.label)):
            Fb = (1 + pow(b, 2)) * (prec[i] * rec[i]) / ((pow(b, 2) * prec[i]) + rec[i])
            f.append(Fb)

        self.printResults("f" + str(b) + "-score", f)
        return f

    def f1score(self):
        """
        Returns the f1 score values of a classifier.

        :param pred: Predicted classification from the classifier.
        :param truth: Ground truth classification
        :param label: List of possible categories.
        :return: a list of the f1 scores of each category.
        """
        return self.fScore(1)

    def printResults(self, type, results):
        """
        Called by a loss function method to print out it's results

        :param type: A string of the name of the loss function
        :param results: A Python List of results of the loss function
        """
        out = "The " + type + " for: "
        for i, result in enumerate(results):
            out += "Category: " + str(self.label[i]) + " is " + str(result) + " "
        print(out)
