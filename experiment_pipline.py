import numpy as np
import pandas as pd
from stratification import *
from KNN import *
from preprocess import *
from evaluation import *
import warnings
warnings.filterwarnings("ignore")

class experiment_pipeline:
    # clean, Split into train/test and Tune, 
    # Tuning method
    # cross validation method
    
    # self.tune_df is df used for tuning (10% of orginal df)
    # self.data is the other 90%
    # self.stratified_data is arrary of ten folds
    
    
    
    def clean(self,data,index,name,isClassification):
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
        prePro = Preprocessor(data,index,name)
        prePro.removemissingvalues() 
        data = prePro.df
        
        # split into train/test and tune df's
        self.tune_df = data.sample(frac=0.1,random_state=200)
        self.data = data.drop(tune_df.index)
        
        # initialize stratification object
        st = strat()
        
        if(isClassification):
            self.stratified_data = st.stratification(data,index,nFold)
        else:
            self.stratified_data = st.stratification_regression(data,index,Nfold)
        
    
    def tuning(self,index,isClassification):
        
        # hp k[always odd] range - % of observations, bandwidth(regression - kernel func) range - , error(edited knn)
        # 100 for maxiter
        
        # same k for knn and kmeans:
        
        # sum percision and recall 
        parameter_matrix = []
        
        # Hyperparameter Ranges
        
        k = [1,2,3,4,5]
        bandwidth = [1,2,3,4,5]
        
        knn_model = KNN(self.data,self.data.columns[0:index],self.data.columns[index],self.tune_df.columns[0:index],self.tune_df.columns[index],index)
        
        for k in k_range:
            if(isClassification):
                # Knn
                prediction = knn_model.knnRegular(k,isClassification,bandwidth)
                ev = Evaluation(prediction[0],prediction[1],self.tune_df.columns[index] )
                parameter_matrix.append([k,(ev.precision+ev.recall)])
            else:
                for bdn in bandwidth:
                    prediction = knn_model.knnRegular(k,isClassification,bnd)
                    ev = Evaluation(prediction[0],prediction[1],self.tune_df.columns[index])
                    parameter_matrix.append([k,bnd,(ev.precision+ev.recall)])
         
         
                    
                              
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
    