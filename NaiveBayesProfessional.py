import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def driver(self,training,train_y,test,test_y):
        """
        mini driver to call learn and predict function

        Parameters
        ----------
        training : Training dataset (x) DataFrame
        train_y : Class(y) corresponding to training dataset DataFrame
        test : Test dataset (x) DataFrame
        test_y : Class(y) corresponding to testing dataset DataFrame

        Returns
        -------
        pred_and_groundTruth :Array of 2 lists where the first list is algorithums predictions and the second is actual observerd class. 

        """
        
        pred_and_groundTruth = []
        
        # Turn test_y into list from dataframe
        true_class = []
        for i in test_y:
            true_class.append(i)

        self.learn(training,train_y)
        
        pred_and_groundTruth.append(self.predict(test))
        pred_and_groundTruth.append(true_class)
        
        return pred_and_groundTruth
         
    def learn(self,data,response):   
        """
        Learns meta information about training dataset

        Parameters
        ----------
        data : Training dataset (x) DataFrame
        response : Class(y) corresponding to training dataset DataFrame

        Returns
        -------
        None.

        """
        # meta information of data
        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        self.num_classes = list(np.unique(response))
            
        # Count of each class
        self.cls_dict = {}
        for cls in self.num_classes:
            self.cls_dict.update({cls: np.count_nonzero(response == cls)})
        
        self.feature_likelyhood = self.calc_feature_likelyhood(data,response)
        print(self.feature_likelyhood)
        
    def predict(self,data):
        """
        Returns predictions for test dataset

        Parameters
        ----------
        data : Test dataset(x) Datafrane

        Returns
        -------
        predictions : List of predicted classes based on input features

        """
        alpha = 1
        predictions = []
         
        for i in data.iterrows():    # Iterate through rows
            tmp = []
            for cls in range(len(self.num_classes)): # Iterate through number of classes in dataset
                
                feature_index = 0
                probability = 1
                               
                for feature in i[1]:        # check if feature is in likelyhood dictionary
                    if self.feature_likelyhood[cls][feature_index].get(feature) is None:
                        probability *= alpha / ( self.cls_dict.get(self.num_classes[cls]) + alpha * len(self.num_classes))  # laplace smoothing
                    else:
                        probability *= self.feature_likelyhood[cls][feature_index].get(feature)
                    
                    feature_index +=1
                    
                tmp.append((self.cls_dict.get(self.num_classes[cls]) / self.num_samples)  * probability)
                
            predictions.append(self.num_classes[tmp.index(max(tmp))]) 
        
        return predictions
        
        
    def calc_feature_likelyhood(self,data,response):     
        """
        Calculates feature probability matrix of training data with laplace smoothing

        Parameters
        ----------
        data : Training dataset(x) DataFrame
        response : Class(y) corresponding to training dataset DataFrame

        Returns
        -------
        likelyHood_array : List of n probability matricies. where n is number of classes 

        """
        alpha = 1
        split_data = self.calc_lk(data,response)
        likelyHood_array = []
        
        for s in split_data: # array s in split data 
            tmp = []
            for i in s:    # columns in s 
                
                likelyhood = {}
                for j,value in s[i].items():   # Iterates through col and counts number of unique values
                    
                    if value in likelyhood.keys():                      
                        likelyhood[value] += 1
                    else:
                        likelyhood[value] = 1 


                for count in likelyhood.keys(): # Iterates through dictionary keys and divides by num_observations
                    likelyhood[count] = (likelyhood.get(count) + alpha) / (s.shape[0] +alpha * len(self.num_classes))         # laplace smoothing 
                
                tmp.append(likelyhood)
            likelyHood_array.append(tmp)
        return likelyHood_array

    def calc_lk(self,data,response):
        """
        Splits trainging dataset by class. Helper function to calc_feature_likelyhood. 

        Parameters
        ----------
        data : Training dataset(x) DataFrame
        response : Class(y) corresponding to training dataset DataFrame

        Returns
        -------
        list of n DataFrames. Where n is number of classes.

        """
        
        split_data = []
        
        for x in range(len(self.num_classes)): # adds n number of DataFrames to split_data 
            
            split_data.append(pd.DataFrame())
        
        ind_count = 0
        # print(response)
        for cls in self.num_classes:            # iterates through number of classes and adds individual data points to correct dataframe in split_data
            tmp = []
            for i,y in enumerate(response):
                if y == cls:
                    tmp.append(data.iloc[i])
                    
            split_data[ind_count] = pd.DataFrame(tmp)
            ind_count +=1
        return split_data
    