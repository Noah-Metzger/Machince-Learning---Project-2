from preprocess import *
from evaluation import *
from execute import *
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
    preProcessedArray = []
    breastCancer = pd.read_csv("Data/breast-cancer-wisconsin.csv", header=None)
    glass = pd.read_csv('Data/glass.csv', header=None)
    houseVotes = pd.read_csv('Data/house-votes-84.csv', header=None)
    iris = pd.read_csv('Data/iris.csv', header=None)
    soyBean = pd.read_csv('Data/soybean-small.csv', header=None)

    #Each dataset is put into a Preprocessor object before classification
    #A dataset is not preprocessed unless a method has been explicit called on the Preprocessor object.
    #The creation of a preprocessor object does not imply that the dataset has been modified in any way.

    #Breast Cancer dataset without noise
    breastCancerNoNoise = Preprocessor(copy.copy(breastCancer), 10, "Breast Cancer Wisconsin")
    breastCancerNoNoise.deleteFeature(0)
    breastCancerNoNoise.removesmissingvalues()
    preProcessedArray.append(breastCancerNoNoise)

    # Breast Cancer dataset with noise
    breastCancerNoise = Preprocessor(copy.copy(breastCancer), 10, "Breast Cancer Wisconsin - Noise Introduced")
    breastCancerNoise.deleteFeature(0)
    breastCancerNoise.removesmissingvalues()
    breastCancerNoise.shuffle()
    preProcessedArray.append(breastCancerNoise)

    #Tuning hyper-parameter for number of bins for glass dataset
    arr1 = []
    for i in range(7,22):
        glassTune = Preprocessor(copy.copy(glass), 10, "Glass Tuning")
        glassTune.deleteFeature(0)
        arr1.append(glassTune)
    tune(arr1, 3, list(range(1, 9)))

    #Glass dataset without noise
    glassNoNoise = Preprocessor(copy.copy(glass), 10, "Glass")
    glassNoNoise.deleteFeature(0)
    glassNoNoise.binning(list(range(1, 9)), 7)
    preProcessedArray.append(glassNoNoise)

    #Glass dataset with noise
    glassNoise = Preprocessor(copy.copy(glass), 10, "Glass - Noise Introduced")
    glassNoise.deleteFeature(0)
    glassNoise.binning(list(range(1, 9)), 7)
    glassNoise.shuffle()
    preProcessedArray.append(glassNoise)

    #House votes dataset without noise
    houseVotesNoNoise = Preprocessor(copy.copy(houseVotes), 0, "House Votes 84")
    preProcessedArray.append(houseVotesNoNoise)

    #House votes dataset with noise
    houseVotesNoise = Preprocessor(copy.copy(houseVotes), 0, "House Votes 84 - Noise Introduced")
    houseVotesNoise.shuffle()
    preProcessedArray.append(houseVotesNoise)

    #Tuning hyper-parameter for number of bins for Iris dataset
    arr = []
    for i in range(3,10):
        irisTune = Preprocessor(copy.copy(iris), 4, "Iris Tuning")
        arr.append(irisTune)
    tune(arr, 3, [0, 1, 2, 3])

    #Iris dataset without noise
    irisNoNoise = Preprocessor(copy.copy(iris), 4, "Iris")
    irisNoNoise.binning([0, 1, 2, 3], 6)
    preProcessedArray.append(irisNoNoise)

    #Iris dataset with noise
    irisNoise = Preprocessor(copy.copy(iris), 4, "Iris - Noise Introduced")
    irisNoise.binning([0, 1, 2, 3], 6)
    irisNoise.shuffle()
    preProcessedArray.append(copy.copy(irisNoise))

    #Soy bean dataset without noise
    soyBeanNoNoise = Preprocessor(copy.copy(soyBean), 35, "Soy bean")
    preProcessedArray.append(soyBeanNoNoise)

    #Soy bean dataset with noise
    soyBeanNoise = Preprocessor(copy.copy(soyBean), 35, "Soy bean - Noise Introduced")
    soyBeanNoise.shuffle()
    preProcessedArray.append(soyBeanNoise)

    #Passes array of all 10 different datasets to run experiment
    experiment(preProcessedArray)