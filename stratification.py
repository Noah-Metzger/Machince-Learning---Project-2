import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class strat:
    
    def stratification(self,data,index,nFold):
        """
        Returns a stratified dataset with nfold where folds class distribution will represent the overall data class distribution

        Parameters
        ----------
        data : cleaned data to be stratified
        index : Index of response(groundTruth) column
        nFold : n number of folds

        Returns
        -------
        stratified_data : stratified Dataframe with nfold number of folds

        """
        
        stratified_data = []
        split_data = self.split(data,index)
        
        for i,df in enumerate(split_data): # Splits each class dataset into Nfolds
            split_data[i] = np.array_split(df, nFold)
        
        for x in range(nFold):# append epmty df for each fold
            stratified_data.append(pd.DataFrame()) # append epmty df for each fold
           
        for z,df in enumerate(stratified_data): # add data for each fold and class 
            tmp = pd.DataFrame()
            
            for y in split_data:
                
                tmp = tmp.append(y[z])
            
            stratified_data[z] = tmp

        # print("Stratified Data")
        # print(stratified_data)

        return stratified_data
        
    def stratification_regression(self,data,index,nFold):
        """
        Returns Stratified Dataset if nfold number of folds for regression datasets

        Parameters
        ----------
        data : cleaned data to be stratified
        index : Index of response(groundTruth) column
        nFold : n number of folds

        Returns
        -------
        strat_reg_data : stratified Dataframe with nfold number of folds

        """
        
        tmp = [] # holds arranged data in partitions of 10 observations
        strat_reg_data = [] # holds final ten folds
        
        data = data.sort_values(data.columns[index]) # sort by response column
        
        num_samples = data.shape[0]
        
        # split data into m number of 10 samples where m is number of samples divided by 10
        tmp = np.array_split(data, num_samples/nFold)
        
        # add nfold number of dataframes to strat
        for x in range(nFold):
            strat_reg_data.append([]) # append epmty df for each fold
        
        # iterate through all partitions in tmp 
        for partition in tmp:
            
            for ind,fld in enumerate(strat_reg_data): # Iterates through nfold number of folds in strat_reg
                
                fld = fld.append(partition.iloc[[ind]])
                
        return strat_reg_data
            
            
    def split(self,data,index):
        """
        Splits dataset into dataframes by classes

        Parameters
        ----------
        data : cleaned data to be split
        index : index of response columns

        Returns
        -------
        split_data : array of n number of dataframes where n is number of unique classes

        """
        
        self.num_classes = list(np.unique(data.iloc[:, index]))
        
        split_data = []
        
        for x in range(len(self.num_classes)): # adds n number of DataFrames to split_data 
            
            split_data.append(pd.DataFrame())
        
        ind_count = 0
        
        for cls in self.num_classes:     # iterates through number of classes and adds individual data points to correct dataframe in split_data
            tmp = []
            for i,y in enumerate(data.iloc[:, index]):
                if y == cls:
                    tmp.append(data.iloc[i])
                    
            split_data[ind_count] = pd.DataFrame(tmp)
            ind_count +=1
        
        return split_data