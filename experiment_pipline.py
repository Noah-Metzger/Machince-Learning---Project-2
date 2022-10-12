import numpy as np
import pandas as pd
from stratification import *
from KNN import *
from preprocess import *
from evaluation import *
import warnings
warnings.filterwarnings("ignore")

class experiment_pipeline:
    def __init__(self, data, isClassification, index, nFold, prePro):
        self.data = data
        self.isClassification = isClassification
        self.index = index
        self.nFold = nFold
        self.num_classes = len(np.unique(prePro.df[prePro.df.columns[index]]))
        self.Preprocessor = prePro

        
    # clean, Split into train/test and Tune, 
    # Tuning method
    # cross validation method
    
    # self.tune_df is df used for tuning (10% of orginal df)
    # self.data is the other 90%
    # self.stratified_data is arrary of ten folds
    
    
    
    def clean(self):
        """
        clean's rxaw data file, splits data into to tune & test/train of 10/ 90 percent. Builds a stratified dataset

        Parameters
        ----------
        data : raw data
        index : index of groundtruth column
        name : name of file
        isClassification : Boolean: True => classification ; False => Regression

        Returns
        -------
        None.

        """     
        # remove missing values
        cln_data = self.data

        # split into train/test and tune df's
        self.tune_df = cln_data.sample(frac=0.1,random_state=200)
        self.cln_data = cln_data.drop(self.tune_df.index)

        # split into train_x, train_y, test_x, test_y for tuning process
        self.train_x = self.cln_data.drop(self.cln_data.columns[self.index],axis = 1) ### watch out for zero!!!!!!!!!!
        self.train_y = self.cln_data[self.cln_data.columns[self.index]]
        self.test_x = self.tune_df.drop(self.tune_df.columns[self.index],axis = 1) ### watch out for zero!!!!!!!!!!!!
        self.test_y = self.tune_df[self.cln_data.columns[self.index]]
        
        # initialize stratification object
        st = strat()

        if(self.isClassification):
            self.stratified_data = st.stratification(self.cln_data,self.index,self.nFold)
        else:
            self.stratified_data = st.stratification_regression(self.cln_data,self.index,self.nFold)
        
    
    def tuning(self):
        """
        Tunes hyperparameter k for knn classification and k & bandwidth for regression

        Returns
        -------
        Array of parameters if classification returns k; if regression, returns k & bandwidth

        """
        parameter_matrix = []
        
        # Hyperparameter Ranges
        k_range = [1,3,5,7,9]
        bandwidth = [0.25,0.5,1,5,10]        
        knn_model = KNN(self.cln_data,self.train_x,self.train_y,self.test_x,self.test_y, self.index)
        
        for k in k_range:
            
            if(self.isClassification):
                # Knn
                prediction = knn_model.knnRegular(k,self.isClassification,bandwidth)
                
                # Evaluate Performance
                ev = Evaluation(prediction[0],prediction[1],self.cln_data[self.cln_data.columns[self.index]] )
                precision_list = ev.precision()
                recall_list = ev.recall()
                
                # Average Percsion and Recall across groundtruths
                precision = 0
                recall = 0
                for ind in range(len(precision_list)):
                    precision += precision_list[ind]
                    recall += recall_list[ind]
                
                precision = precision /len(precision_list)
                recall = recall /  len(recall_list)
                
                # append to matrix
                parameter_matrix.append([precision+recall,k])
            else:
                for bnd in bandwidth:
                    #print(bnd,k)   
                    prediction = knn_model.knnRegular(k,self.isClassification,bnd)
                    # Evaluate Performance
                    ev = Evaluation(prediction[0],prediction[1],self.cln_data[self.cln_data.columns[self.index]] )
                    parameter_matrix.append([ev.RelativeAbsoluteError(),k,bnd])
        
        # find index of best performing hyperparameter
        tmp = []
        for g in parameter_matrix:
            tmp.append(g[0])
                 
        max_index = np.argmax(tmp)
        min_index = np.argmin(tmp)
        if(self.isClassification):
            return(parameter_matrix[max_index][1])
        else:
            return(parameter_matrix[min_index][1],parameter_matrix[min_index][2])
                     

    def editedknn_tuning(self):
        """
        Tunes hyperparameter error. Finds error value that returns a reduced dataset closest to 90% of orignal data

        Parameters
        ----------
        Returns
        -------
        TYPE
            best performing value of error

        """

        err = 0
        k_range = [1,3,5,7,9]
        bandwidth = [0.25,0.5,1,5,10]
        error = [0.25, 0.5, 1, 10]
        
        knn_model = KNN(self.cln_data,self.train_x,self.train_y,self.test_x,self.test_y,self.index)
        
        length_vec = []
        for k in k_range:
            for bnd in bandwidth:
                if not self.isClassification:
                    for err in error:
                        df = knn_model.knnEdited(k, self.isClassification, bnd, err)
                        results = self.crossvalidationknnregularReduced(k, bnd, df)
                        preformance = []
                        for fold in results:
                            e = Evaluation(fold[0], fold[1], self.data[self.data.columns[self.index]])
                            preformance.append(e.MeanAbsoluteError() + e.RelativeAbsoluteError())
                        length_vec.append([(sum(preformance)/len(preformance)), k,bnd,err])
                else:
                    df = knn_model.knnEdited(k, self.isClassification, bnd, -1)
                    results = self.crossvalidationknnregularReduced(k, bnd, df)
                    preformance = []
                    for fold in results:
                        e = Evaluation(fold[0], fold[1], self.data[self.data.columns[self.index]])
                        prec = e.precision()
                        rec = e.recall()
                        preformance.append((sum(prec) / len(prec)) + (sum(rec) / len(rec)))
                    length_vec.append([(sum(preformance) / len(preformance)), k, bnd, -1])
        
        # find index of best performing hyperparameter
        tmp = []
        for g in length_vec:
            tmp.append(g[0])
        index = 0
        if self.isClassification:
            index = np.argmax(tmp)
        else:
            index = np.argmin(tmp)
        
        return (length_vec[index][1],length_vec[index][2],length_vec[index][3])
    
    def crossvalidationknnregular(self, k, bandwidth):
        """
        Normal KNN 10 fold cross validation.

        :param k: K number of neighbors to be used.
        :param bandwidth: Bandwidth of kernel function.
        :return: Array of predicted values/classes and truth values/classes
        """
        results = []

        #Iterates through each fold
        for i in range(len(self.stratified_data)):
            #Separates dataset into training and test datasets
            train = pd.DataFrame()
            test = pd.DataFrame()
            offset = 0
            for j, fold in enumerate(self.stratified_data):
                if j + offset == i:
                    if len(fold) == 0:
                        offset += 1
                    else:
                        test = test.append(fold)
                else:
                    train = train.append(fold)

            # if i == 0:
            #     print("10 fold cross validation")
            #     print("test")
            #     print(test)
            #     print("train")
            #     print(train)

            #Separates ground truth column from training and test set
            train_response = train.iloc[:, self.index]
            train.drop(self.data.columns[[self.index]], axis=1, inplace=True)

            test_response = test.iloc[:, self.index]
            test.drop(self.data.columns[[self.index]], axis=1, inplace=True)

            knn = KNN(self.stratified_data, test, test_response, train, train_response, self.index)
            results.append(knn.knnRegular(k, self.isClassification, bandwidth))

        return results

    def crossvalidationknnregularReduced(self, k, bandwidth, full):
        """
        Normal KNN 10-fold cross validation except dataset is inputed from a parameter not self.

        :param k: K number of neighbors to be used.
        :param bandwidth: Bandwidth of kernel function.
        :param full: Dataset to have 10-fold cross validation ran on
        :return: Array of predicted values/classes and truth values/classes
        """
        results = []

        folds = np.array_split(full, 10)

        #Iterates through each fold
        for i in range(len(folds)):
            #Separates dataset into training and test datasets
            train = pd.DataFrame()
            test = pd.DataFrame()
            for j, fold in enumerate(folds):
                if j == i:
                    if len(fold) == 0:
                        return results
                    else:
                        test = test.append(fold)
                else:
                    train = train.append(fold)

            #Separates ground truth column from training and test set
            train_response = train.iloc[:, self.index]
            train.drop(self.data.columns[[self.index]], axis=1, inplace=True)

            test_response = test.iloc[:, self.index]
            test.drop(self.data.columns[[self.index]], axis=1, inplace=True)

            knn = KNN(full, test, test_response, train, train_response, self.index)
            results.append(knn.knnRegular(k, self.isClassification, bandwidth))

        return results


    def editedknn(self, k, bandwidth, error):
        """
        Function configuring parameters for edited KNN

        :param k: K number of neighbors to be used.
        :param bandwidth: Bandwidth of kernel function
        :param error: Maximum error a predicted value can be from the truth value to be considered correct (does not effect classification)
        :return: The edited-KNN reduced dataset
        """
        full = pd.DataFrame()
        for j, fold in enumerate(self.stratified_data):
            full = full.append(fold)
        knn = KNN(full, 0, 0, 0, 0, self.index)
        result = knn.knnEdited(k, self.isClassification, bandwidth, error)
        return result

    def kMeans(self, k, isClassification):
        """
        K-Means clustering algorithm

        :param k: K number of neighbors to be used.
        :param isClassification: Boolean whether or not problem is classification problem
        :return: Array of k-Means cluster centroids
        """
        full = pd.DataFrame()
        for j, fold in enumerate(self.stratified_data):
            full = full.append(fold)

        kMeans = KNN(full, 0,0,0,0,self.index)
        return kMeans.Kmeans(k, 100, isClassification)