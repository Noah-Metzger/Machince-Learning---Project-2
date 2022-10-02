import numpy as np
import pandas as pd
from stratification import *
from KNN import *
from preprocess import *
from evaluation import *
import warnings
warnings.filterwarnings("ignore")

class experiment_pipeline:
    
    def __init__(self, data, isClassification, index, nFold):
        
        self.data = data
        self.isClassification = isClassification
        self.index = index
        self.nFold = nFold
        
        
    # clean, Split into train/test and Tune, 
    # Tuning method
    # cross validation method
    
    # self.tune_df is df used for tuning (10% of orginal df)
    # self.data is the other 90%
    # self.stratified_data is arrary of ten folds
    
    
    
    def clean(self,name):
        """
        clean's raw data file, splits data into to tune & test/train of 10/ 90 percent. Builds a stratified dataset

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
        prePro = Preprocessor(self.data,self.index,name)
        prePro.removesmissingvalues() 
        cln_data = prePro.df
               
        # split into train/test and tune df's
        self.tune_df = cln_data.sample(frac=0.1,random_state=200)
        self.cln_data = cln_data.drop(self.tune_df.index)
        
        # split into train_x, train_y, test_x, test_y for tuning process
        self.train_x = self.cln_data.drop(self.cln_data.columns[[0,self.index]],axis = 1) ### watch out for zero!!!!!!!!!!
        self.train_y = self.cln_data[self.index]
        self.test_x = self.tune_df.drop(ep.tune_df.columns[[0, self.index]],axis = 1) ### watch out for zero!!!!!!!!!!!!
        self.test_y = self.tune_df[self.index]
        
        # initialize stratification object
        st = strat()
        
        if(self.isClassification):
            self.stratified_data = st.stratification(self.cln_data,self.index,self.nFold)
        else:
            self.stratified_data = st.stratification_regression(self.cln_data,self.index,self.nFold)
        
    
    def tuning(self):
        
        # hp k[always odd] range - % of observations, bandwidth(regression - kernel func) range - , error(edited knn)
        # 100 for maxiter
        
        # same k for knn and kmeans:
        
        # sum percision and recall 
        parameter_matrix = []
        
        # Hyperparameter Ranges
        
        k_range = [1,2,3,4,5]
        bandwidth = [1,2,3,4,5]
        
        knn_model = KNN(self.cln_data,self.train_x,self.train_y,self.test_x,self.test_y,self.index)
        
        for k in k_range:
            
            if(self.isClassification):
                # Knn
                prediction = knn_model.knnRegular(k,self.isClassification,bandwidth)
                
                # Evaluate Performance
                ev = Evaluation(prediction[0],prediction[1],self.cln_data[self.index] )
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
                    print(bnd)
                       
                    prediction = knn_model.knnRegular(k,self.isClassification,bnd)
                    # Evaluate Performance
                    ev = Evaluation(prediction[0],prediction[1],self.cln_data[index] )
                                               
                    parameter_matrix.append([ev.getAverageError(),k,bnd])
        
        # find index of best performing hyperparameter
        tmp = []
        for g in parameter_matrix:
            tmp.append(g[0])
            
            
        max_index = np.argmax(tmp)
        
        if(self.isClassification):
            
            return(parameter_matrix[max_index][1])
        else:
            return(parameter_matrix)
                              
    def crossvalidation(self,func,index):
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
            train_response = train.iloc[:, index]
            train.drop(index, axis=1, inplace=True)

            test_response = test.iloc[:, index]
            test.drop(index, axis=1, inplace=True)

            results.append(func(train, train_response, test, test_response))

        return results
    
data = pd.read_csv("Data/breast-cancer-wisconsin.csv",header=None)

ep = experiment_pipeline(data,True,10,10)

ep.clean("name")

#index = 8  
#kn = KNN(ep.cln_data,ep.cln_data.drop(ep.cln_data.columns[[0, 8]],axis = 1),ep.cln_data[index],ep.tune_df.drop(ep.tune_df.columns[[0, 8]],axis = 1),ep.tune_df[index],index)
#kn.knnRegular(5, False, 2)

print(ep.tuning())
    