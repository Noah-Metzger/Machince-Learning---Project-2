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
        cln_data = self.Preprocessor.df

        # split into train/test and tune df's
        self.tune_df = cln_data.sample(frac=0.1,random_state=200)
        self.cln_data = cln_data.drop(self.tune_df.index)

        # split into train_x, train_y, test_x, test_y for tuning process
        self.train_x = self.cln_data.drop(self.cln_data.columns[[0,self.index]],axis = 1) ### watch out for zero!!!!!!!!!!
        self.train_y = self.cln_data[self.cln_data.columns[self.index]]
        self.test_x = self.tune_df.drop(self.tune_df.columns[[0, self.index]],axis = 1) ### watch out for zero!!!!!!!!!!!!
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
        k_range = []
        input_k = 0 
        for hp_k in range(5):
            if self.num_classes % 2 == 0 and  hp_k == 0:
                input_k = self.num_classes +1
                k_range.append(input_k)
            
            else:
                input_k += 2
                k_range.append(input_k)
            
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
                    parameter_matrix.append([ev.MeanAbsoluteError() + ev.RelativeAbsoluteError(),k,bnd])
        
        # find index of best performing hyperparameter
        tmp = []
        for g in parameter_matrix:
            tmp.append(g[0])
                 
        max_index = np.argmax(tmp)
        min_index = np.argmin(min)
        
        if(self.isClassification):
            return(parameter_matrix[max_index][1])
        else:
            return(parameter_matrix[min_index][1],parameter_matrix[min_index][2])
                     

    def editedknn_tuning(self,k,bnd):
        """
        Tunes hyperparameter error. Finds error value that returns a reduced dataset closest to 90% of orignal data

        Parameters
        ----------
        k : k found from tunning()
        bnd : bandwidth found from tunning() for regression problems

        Returns
        -------
        TYPE
            best performing value of error

        """
        std = np.std(self.train_y)
        print(std)
        error_vector = [(.1 * std),(.25 * std),(.5 * std),(.75 * std),std]
        
        knn_model = KNN(self.cln_data,self.train_x,self.train_y,self.test_x,self.test_y,self.index)
        
        length_vec = []
        for err in error_vector:
            df = knn_model.knnEdited(k, self.isClassification, bnd, err)
            length_vec.append([((df.shape[0] /self.data.shape[0]) - .9) , err])
        
        # find index of best performing hyperparameter
        tmp = []
        for g in length_vec:
            tmp.append(g[0])        
        min_index = np.argmin(min)
        
        return length_vec[min_index][1]
    
    def crossvalidationknnregular(self, k, bandwidth):
        results = []

        #Iterates through each fold
        for i in range(len(self.stratified_data)):
            #Separates dataset into training and test datasets
            train = pd.DataFrame()
            test = pd.DataFrame()
            for j, fold in enumerate(self.stratified_data):
                if j == i:
                    test = test.append(fold)
                else:
                    train = train.append(fold)

            #Separates ground truth column from training and test set
            train_response = train.iloc[:, self.index]
            train.drop(self.index, axis=1, inplace=True)

            test_response = test.iloc[:, self.index]
            test.drop(self.index, axis=1, inplace=True)

            knn = KNN(self.stratified_data, test, test_response, train, train_response, self.index)
            results.append(knn.knnRegular(k, self.isClassification, bandwidth))

        return results

    def editedknn(self, k, bandwidth, error):
        full = pd.DataFrame()
        for j, fold in enumerate(self.stratified_data):
            full = full.append(fold)
        knn = KNN(full, 0, 0, 0, 0, self.index)
        result = knn.knnEdited(k, self.isClassification, bandwidth, error)
        return result

    def kMeans(self, k):
        full = pd.DataFrame()
        for j, fold in enumerate(self.stratified_data):
            full = full.append(fold)
        kMeans = KNN(full, 0,0,0,0,self.index)
        return kMeans.Kmeans(k, 100)





# data = pd.read_csv("Data/breast-cancer-wisconsin.csv",header=None)
#
# ep = experiment_pipeline(data,False,10,10)
#
# ep.clean("name")
#
# #index = 8
# #kn = KNN(ep.cln_data,ep.cln_data.drop(ep.cln_data.columns[[0, 8]],axis = 1),ep.cln_data[index],ep.tune_df.drop(ep.tune_df.columns[[0, 8]],axis = 1),ep.tune_df[index],index)
# #kn.knnRegular(5, False, 2)
# print(ep.editedknn_tuning(3, .25))
# #ep.editedknn_tuning()
#