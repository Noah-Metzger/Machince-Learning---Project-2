from preprocess import *
from evaluation import *
from execute import *
from KNN import *
from NaiveBayesProfessional import *
import copy
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")

def tune(prepro, lowerBound, cols):
    """
    Prints the results of average precision and recall values for each bin. Prints out sorted list of values and the number of bins for each corresponding value.

    :param prepro: Preprocessor object for dataset for number of bins to be tuned for
    :param lowerBound: The lower bound of number of bins to test.
    :param upperBound: The upper bound of number of bins to test.
    :param cols: Array of column indexes to bin.
    """
    tuning = []
    for b in range(len(prepro)):
        #Binning of dataset for each number in the range
        prepro[b].binning(cols, b+lowerBound)
        # Conducts classification with 10-fold cross-validation
        n = NaiveBayesClassifier()
        e = execute(prepro[b].df)
        results = e.crossvalidate(n.driver, 10, prepro[b].truthColIndex)

        x = []
        y = []
        # Prints out precision and recall for each fold
        for fold in results:
            e = Evaluation(fold[0], fold[1], np.array(prepro[b].truthCol))
            prec = e.precision()
            rec = e.recall()
            for i in range(len(prec)):
                x.append(prec[i])
                y.append(rec[i])

        tuning.append((sum(x) / len(x)) + (sum(y) / len(y)))

    #Creating a list of indexes to track the positions of values in tuning array
    index = list(range(0,len(tuning)))

    #Bubble sorts values in tuning array, when values are swapped in tuning array, the same indices are swapped index array to keep track of values position.
    for i in range(len(tuning)):
        for j in range(0, len(tuning) - i - 1):
            if tuning[i] > tuning[j]:
                temp = tuning[i]
                tuning[i] = tuning[j]
                tuning[j] = temp

                tempIndex = index[i]
                index[i] = index[j]
                index[j] = tempIndex

    #Adds the lower bound of range of number of bins to check to accurately print number of bins with corresponding preformance.
    for i in range(len(index)):
        index[i] = index[i] + lowerBound
    #Prints out tuning results
    print(tuning)
    print(index)
    print(str(index[0]) + " number of bins has the greatest (precison + recall)")

def plot(result1, result2, name):
    """
    Outputs a scatter plot of results from two loss-functions

    :param result1: Python list of values to be plotted on the x-axis.
    :param result2: Python list of values to be plotted on the y-axis.
    :param name: Name of the dataset to be used as title for the plot
    """
    plt.scatter(result1, result2)
    plt.title(name + " Dataset")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()
    # plt.savefig(name + ".png")

def experiment(preproArr):
    """
    The main driver for the experimentation of the classifier.  Runs classifier, k-fold, cross validation, evaluations, printing and plotting of results

    :param preproArr: A Python list of Preprocessor objects.  These do not necessarily need to have any preprocessing methods called before classification
    """
    for obj in preproArr:
        #Conducts classification with 10-fold cross-validation
        n = NaiveBayesClassifier()
        e = execute(obj.df)
        results = e.crossvalidate(n.driver, 10, obj.truthColIndex)
        x = []
        y = []
        #Prints out precision and recall for each fold
        print("***" + obj.dfName + "***")
        for fold in results:
            print(fold[0])
            e = Evaluation(fold[0], fold[1], np.array(obj.truthCol))
            prec = e.precision()
            rec = e.recall()
            # e.printConfusionMatrix()
            for i in range(len(prec)):
                x.append(prec[i])
                y.append(rec[i])

        print()
        print("Average precision: " + str(sum(x) / len(x)))
        print("Average recall: " + str(sum(y) / len(y)))
        print()
        plot(np.array(x), np.array(y), obj.dfName)

    #Prints scatter plots

if __name__ == '__main__':
    exPipe = []
    abalone = pd.read_csv("Data/abalone.csv", header=None)
    breastCancer = pd.read_csv("Data/breast-cancer-wisconsin.csv", header=None)
    forestFire = pd.read_csv("Data/forestfires.csv")
    glass = pd.read_csv('Data/glass.csv', header=None)
    machine = pd.read_csv('Data/machine.csv', header=None)
    soyBean = pd.read_csv('Data/soybean-small.csv', header=None)

    abalonePre = Preprocessor(copy.copy(abalone), 8, "Abalone")
    abalonePre.onehotencoding([0])
    abaloneEx = experiment_pipline(abalonePre.df, False, abalonePre.truthColIndex, 10)
    exPipe.append(abaloneEx)

    breastCancerPre = Preprocessor(copy.copy(breastCancer), 10, "Breast Cancer Wisconsin")
    breastCancerPre.removesmissingvalues()
    breastCancerPre.deleteFeature(0)
    breastCancerEx = experiment_pipline(breastCancerPre.df, False, breastCancerPre.truthColIndex, 10)
    exPipe.append(breastCancerEx)

    forestFirePre = Preprocessor(copy.copy(forestFire), 12, "Forest Fires")
    forestFirePre.labelencodeOridinal(2, ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    forestFirePre.labelencodeOridinal(3, ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
    forestFireEx = experiment_pipline(forestFirePre.df, False, forestFirePre.truthColIndex, 10)
    exPipe.append(forestFireEx)

    glassPre = Preprocessor(copy.copy(glass), 10, "Glass")
    glassPre.deleteFeature(0)
    glassEx = experiment_pipline(glassPre.df, False, glassPre.truthColIndex, 10)
    exPipe.append(glassEx)

    machinePre = Preprocessor(copy.copy(machine), 9, "Machine")
    machinePre.deleteFeature(0)
    machinePre.deleteFeature(1)
    machineEx = experiment_pipline(machinePre.df, False, machinePre.truthColIndex, 10)
    exPipe.append(machineEx)

    soyBeanPre = Preprocessor(copy.copy(soyBean), 35, "Soybean")
    soyBeanEx = experiment_pipline(soyBeanPre.df, False, soyBeanPre.truthColIndex, 10)
    exPipe.append(soyBeanEx)

    for exObj in exPipe:
        exObj.clean()