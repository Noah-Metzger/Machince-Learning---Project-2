from preprocess import *
from evaluation import *
from execute import *
from KNN import *
from experiment_pipline import *
import copy
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")

def ScatterPlot(result1, result1Label, result2, result2Label, name):
    """
    Outputs a scatter plot of results from two loss-functions

    :param result1: Python list of values to be plotted on the x-axis.
    :param result2: Python list of values to be plotted on the y-axis.
    :param name: Name of the dataset to be used as title for the plot
    """
    plt.scatter(result1, result2)
    plt.title(name + " Dataset")
    plt.xlabel(result1Label)
    plt.ylabel(result2Label)
    plt.show()
    # plt.savefig(name + ".png")

if __name__ == '__main__':
    exPipe = []
    abalone = pd.read_csv("Data/abalone.csv", header=None)
    breastCancer = pd.read_csv("Data/breast-cancer-wisconsin.csv", header=None)
    forestFire = pd.read_csv("Data/forestfires.csv")
    glass = pd.read_csv('Data/glass.csv', header=None)
    machine = pd.read_csv('Data/machine.csv', header=None)
    soyBean = pd.read_csv('Data/soybean-small.csv', header=None)

    # breastCancerPre = Preprocessor(copy.copy(breastCancer), 10, "Breast Cancer Wisconsin")
    # breastCancerPre.removesmissingvalues()
    # breastCancerPre.deleteFeature(0)
    # print(breastCancerPre.df)
    # breastCancerPre.onehotencodeAll()
    # breastCancerPre.labelencode([89])
    # breastCancerEx = experiment_pipeline(breastCancerPre.df, True, breastCancerPre.truthColIndex, 10, breastCancerPre)
    # exPipe.append(breastCancerEx)

    # abalonePre = Preprocessor(copy.copy(abalone), 8, "Abalone")
    # abalonePre.onehotencoding([0])
    # abaloneEx = experiment_pipeline(abalonePre.df, False, abalonePre.truthColIndex, 10, abalonePre)
    # exPipe.append(abaloneEx)

    # forestFirePre = Preprocessor(copy.copy(forestFire), 12, "Forest Fires")
    # forestFirePre.labelencodeOridinal(2, ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    # forestFirePre.labelencodeOridinal(3, ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
    # forestFireEx = experiment_pipeline(forestFirePre.df, False, forestFirePre.truthColIndex, 10, forestFirePre)
    # exPipe.append(forestFireEx)

    # glassPre = Preprocessor(copy.copy(glass), 10, "Glass")
    # glassPre.deleteFeature(0)
    # glassPre.onehotencodeAll()
    # glassEx = experiment_pipeline(glassPre.df, True, glassPre.truthColIndex, 10, glassPre)
    # exPipe.append(glassEx)

    machinePre = Preprocessor(copy.copy(machine), 9, "Machine")
    machinePre.deleteFeature(0)
    machinePre.deleteFeature(1)
    machineEx = experiment_pipeline(machinePre.df, False, machinePre.truthColIndex, 10, machinePre)
    exPipe.append(machineEx)

    soyBeanPre = Preprocessor(copy.copy(soyBean), 35, "Soybean")
    soyBeanPre.onehotencodeAll()
    soyBeanPre.labelencode([72])
    soyBeanEx = experiment_pipeline(soyBeanPre.df, True, soyBeanPre.truthColIndex, 10, soyBeanPre)
    exPipe.append(soyBeanEx)
#
    for exObj in exPipe:
        print("*****" + exObj.Preprocessor.dfName + "*****")
        #Stratifies datasets into equally balanced folds of different classes
        exObj.clean()
        #Tunes KNN on the original dataset
        regular = exObj.tuning()
        print(regular)
        bandwidth = 0
        if exObj.isClassification:
            k = regular
        else:
            k = regular[0]
            bandwidth = regular[1]

        x = []
        y = []

        #Performs a 10-fold cross validation on the original data
        results = exObj.crossvalidationknnregular(k, bandwidth)
        print(results)

        #Calculates precision and recall scores or mean absolute error and mean relative error for each fold
        for fold in results:
            e = Evaluation(fold[0], fold[1], np.array(exObj.data[exObj.data.columns[exObj.index]]))
            if exObj.isClassification:
                x1 = e.precision()
                y1 = e.recall()
                for i in range(len(x1)):
                    x.append(x1[i])
                    y.append(y1[i])
            else:
                x1 = e.MeanAbsoluteError()
                y1 = e.RelativeAbsoluteError()
                x.append(x1)
                y.append(y1)

        #Prints performance metrics to terminal and creates a scatter plot
        if exObj.isClassification:
            print()
            print("Average precision: " + str(sum(x) / len(x)))
            print("Average recall: " + str(sum(y) / len(y)))
            print()
            print(np.array(x), "Precision", np.array(y), "Recall", exObj.Preprocessor.dfName + " KNN Classification")
            ScatterPlot(np.array(x), "Precision", np.array(y), "Recall", exObj.Preprocessor.dfName + " KNN Classification")
        else:
            print()
            print("Average Mean Absolute Error: " + str(sum(x) / len(x)))
            print("Average Relative Absolute Error: " + str(sum(y) / len(y)))
            print()
            print(np.array(x), "Mean Absolute Error", np.array(y), "Relative Absolute Error",exObj.Preprocessor.dfName + " KNN Regression")
            ScatterPlot(np.array(x), "Mean Absolute Error", np.array(y), "Relative Absolute Error",exObj.Preprocessor.dfName + " KNN Regression")

        #For classification
        if exObj.isClassification:
            #Tunes hyper-parameters for edited-KNN
            hyper = exObj.editedknn_tuning()
            print(hyper)
            editedDataset = exObj.editedknn(hyper[0], hyper[1], hyper[2])

            #Create edited-KNN reduced dataset and runs 10-fold cross validation with KNN on the reduced dataset.
            editedKNN = experiment_pipeline(editedDataset, exObj.isClassification, exObj.index, exObj.nFold, exObj.Preprocessor)
            editedKNN.clean()
            resultsedited = editedKNN.crossvalidationknnregular(hyper[0], hyper[1])

            #Calculates precision and recall scores for the classifier
            xedited = []
            yedited = []
            for fold in resultsedited:
                e = Evaluation(fold[0], fold[1], np.array(exObj.data[exObj.data.columns[exObj.index]]))
                x1 = e.precision()
                y1 = e.recall()
                for i in range(len(x1)):
                    xedited.append(x1[i])
                    yedited.append(y1[i])
            # Prints performance metrics to terminal and creates a scatter plot
            print()
            print("Average Precision: " + str(sum(xedited) / len(xedited)))
            print("Average Recall: " + str(sum(yedited) / len(yedited)))
            print()

            print(np.array(xedited), "Precision: ", np.array(yedited), "Recall: ", editedKNN.Preprocessor.dfName + " KNN Classification - Edited-KNN Reduced")
            ScatterPlot(np.array(xedited), "Precision: ", np.array(yedited), "Recall: ",editedKNN.Preprocessor.dfName + " KNN Classification - Edited-KNN Reduced")

            #Use number of instances in edited-KNN dataset for number of k-clusters
            kClusters = len(editedDataset)
            print(kClusters)
            x = []
            y = []
            #Perform k-Means and return centroids of clusters
            centeriods = exObj.kMeans(kClusters, exObj.isClassification)

            #Use k-Means centroids as dataset and runs 10-fold cross validation with KNN on the reduced dataset.
            reducedDataset = experiment_pipeline(centeriods, exObj.isClassification, exObj.index, exObj.nFold,exObj.Preprocessor)
            reducedDataset.clean()
            results = reducedDataset.crossvalidationknnregular(hyper[0], hyper[1])
            for fold in results:
                e = Evaluation(fold[0], fold[1], np.array(reducedDataset.data[reducedDataset.data.columns[reducedDataset.index]]))
                x1 = e.precision()
                y1 = e.recall()
                for i in range(len(x1)):
                    x.append(x1[i])
                    y.append(y1[i])
            # Prints performance metrics to terminal and creates a scatter plot
            print()
            print("Average Precision: " + str(sum(x) / len(x)))
            print("Average Recall: " + str(sum(y) / len(y)))
            print()

            print(np.array(x), "Precison", np.array(y), "Recall", reducedDataset.Preprocessor.dfName + " KNN Classification - KMeans Clusteriod")
            ScatterPlot(np.array(x), "Precision", np.array(y), "Recall", reducedDataset.Preprocessor.dfName + " KNN Classification - KMeans Clusteriod")
        else:
            # For regression
            # Tunes hyper-parameters for edited-KNN
            hyper = exObj.editedknn_tuning()
            print(hyper)
            editedDataset = exObj.editedknn(hyper[0], hyper[1], hyper[2])
            # Create edited-KNN reduced dataset and runs 10-fold cross validation with KNN on the reduced dataset.
            editedKNN = experiment_pipeline(editedDataset, exObj.isClassification, exObj.index, exObj.nFold, exObj.Preprocessor)
            resultsedited = editedKNN.crossvalidationknnregularReduced(hyper[0], hyper[1], editedDataset)

            print(resultsedited)
            xedited = []
            yedited = []
            # Calculates MAE and RAE for the predictor
            for fold in resultsedited:
                e = Evaluation(fold[0], fold[1], np.array(editedKNN.data[editedKNN.data.columns[editedKNN.index]]))
                x1 = e.MeanAbsoluteError()
                y1 = e.RelativeAbsoluteError()
                xedited.append(x1)
                yedited.append(y1)

            print()
            print("Average Mean Absolute Error: " + str(sum(xedited) / len(xedited)))
            print("Average Relative Absolute Error: " + str(sum(yedited) / len(yedited)))
            print()
            # Prints performance metrics to terminal and creates a scatter plot
            print(np.array(xedited), "Mean Absolute Error", np.array(yedited), "Relative Absolute Error", editedKNN.Preprocessor.dfName + " KNN Regression - Edited-KNN Reduced")
            ScatterPlot(np.array(xedited), "Mean Absolute Error", np.array(yedited), "Relative Absolute Error", editedKNN.Preprocessor.dfName + " KNN Regression - Edited-KNN Reduced")

            # Use number of instances in edited-KNN dataset for number of k-clusters
            kClusters = len(editedDataset)
            print(kClusters)
            x = []
            y = []
            # Perform k-Means and return centroids of clusters
            centeriods = exObj.kMeans(kClusters, exObj.isClassification)

            # Use k-Means centroids as dataset and runs 10-fold cross validation with KNN on the reduced dataset.
            reducedDataset = experiment_pipeline(centeriods, exObj.isClassification, exObj.index, exObj.nFold, exObj.Preprocessor)
            reducedDataset.clean()
            tuned = reducedDataset.tuning()
            results = reducedDataset.crossvalidationknnregular(tuned[0], tuned[1])

            # Calculates MAE and RAE for the predictor
            for fold in results:
                e = Evaluation(fold[0], fold[1], np.array(reducedDataset.data[reducedDataset.data.columns[reducedDataset.index]]))
                x1 = e.MeanAbsoluteError()
                y1 = e.RelativeAbsoluteError()
                x.append(x1)
                y.append(y1)

            # Prints performance metrics to terminal and creates a scatter plot
            print()
            print("Average Mean Absolute Error: " + str(sum(x) / len(x)))
            print("Average Relative Absolute Error: " + str(sum(y) / len(y)))
            print()

            print(np.array(x), "Mean Absolute Error", np.array(y), "Relative Absolute Error", reducedDataset.Preprocessor.dfName + " KNN Regression - KMeans Clusteriod")
            ScatterPlot(np.array(x), "Mean Absolute Error", np.array(y), "Relative Absolute Error", reducedDataset.Preprocessor.dfName + " KNN Regression - KMeans Clusteriod")


