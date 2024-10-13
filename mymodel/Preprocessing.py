from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import numpy as np
import seaborn as sns
import imblearn
from statistics import mode
import pickle as pkl

class Preprocessing():

    def __init__(self):
        self._df = None
        self._train = None
        self._test = None
    
    def SetDataFrame(self, target_variable:str, dataset = None, train_data = None, test_data = None):

        if(dataset is None and (train_data is None or test_data is None)):
            raise TypeError('No data detected')
        
        if(dataset is None):
            self._test = test_data.copy()
            self._train = train_data.copy()
            self._original_train = train_data.copy()
            self._original_test = train_data.copy()

        elif(train_data is None or test_data is None):
            self._df = dataset.copy()
            self._original_df = dataset.copy()

        self._target = target_variable

    def GetDataFrame(self, n=None):
        if(self._df is not None):
            return self._df.head(n) if n is not None else self._df
        
        return self._train, self._test #incomplete
    
    def GetInfo(self):
        if(self._df is not None):
            return self._df.info()
        
        return self._train, self._test #incomplete

    def GetShape(self):
        if(self._df is not None):
            return self._df.shape
        
        return self._train, self._test #incomplete
    
    def GetMissingValues(self):

        missing_values = []

        if(self._df is not None):

            for i in self._df:
                missing_values_count = self._df[i].isna().sum()
                if(missing_values_count > 0):
                    missing_values.append([i, missing_values_count])
            

        if(self._train is not None and self._test is not None):

            for i in self._train:
                missing_values_count = self._train[i].isna().sum()

                if(missing_values_count > 0):
                    missing_values.append([i, missing_values_count])

            for i in self._test:
                missing_values_count = self._test[i].isna().sum()

                if(missing_values_count > 0):
                    missing_values.append([i, missing_values_count])

        return missing_values[0] if len(missing_values) > 0 else None
    
    def Boxplot(self, column_name = None):

        if(column_name is None):
            raise TypeError('No column is selected')

        if(self._df is not None):
            self._df.boxplot(column=column_name)
            plt.show()
        
        if(self._train is not None and self._test is not None):
            temp = pd.concat([self._train, self._test], axis=0)

            temp.boxplot(column=column_name)
            plt.show()
    
    def HandleMissingValues(self, na_type = None, outliers:bool = False):

        if(na_type is None):

            if(self._df is not None):

                for i in self._df:

                    if(outliers is True):
                        self._df[i] = self._df[i].fillna(self._df[i].median())

                    elif(self._df[i].dtype == 'object'):
                        self._df[i] = self._df[i].fillna(dict(self._df[i].mode())[0])

                    else:
                        self._df[i] = self._df[i].fillna(self._df[i].mean())
            
            elif(self._train is not None and self._test is not None):
                
                for i in self._train:

                    if(outliers is True):
                        self._train[i] = self._train[i].fillna(self._train[i].median())

                    elif(self._train[i].dtype == 'object'):
                        self._train[i] = self._train[i].fillna(dict(self._train[i].mode())[0])

                    else:
                        self._train[i] = self._train[i].fillna(self._train[i].mean())
                
                for i in self._test:

                    if(outliers is True):
                        self._test[i] = self._test[i].fillna(self._test[i].median())

                    elif(self._test[i].dtype == 'object'):
                        self._test[i] = self._test[i].fillna(dict(self._test[i].mode())[0])

                    else:
                        self._test[i] = self._test[i].fillna(self._test[i].mean())
        
        else:
            
            if(self._df is not None):

                for i in self._df:

                    if(outliers is True and self._df[i].dtype != 'object'):
                        self._df[i] = self._df[i].replace(na_type, np.NaN)
                        self._df[i] = self._df[i].fillna(self._df[i].median())

                    elif(self._df[i].dtype == 'object'):
                        self._df[i] = self._df[i].replace(na_type, np.NaN)
                        self._df[i] = self._df[i].fillna(dict(self._df[i].mode())[0])

                    else:
                        self._df[i] = self._df[i].replace(na_type, np.NaN)
                        self._df[i] = self._df[i].fillna(self._df[i].mean())
            
            elif(self._train is not None and self._test is not None):
                
                for i in self._train:

                    if(outliers is True):
                        self._train[i] = self._train[i].replace(na_type, np.NaN)
                        self._train[i] = self._train[i].fillna(self._train[i].median())
                    
                    elif(self._train[i].dtype == 'object'):
                        self._train[i] = self._train[i].replace(na_type, np.NaN)
                        self._train[i] = self._train[i].fillna(dict(self._train[i].mode())[0])

                    else:
                        self._train[i] = self._train[i].replace(na_type, np.NaN)
                        self._train[i] = self._train[i].fillna(self._train[i].mean())
                
                for i in self._test:

                    if(outliers is True):
                        self._test[i] = self._test[i].replace(na_type, np.NaN)
                        self._test[i] = self._test[i].fillna(self._test[i].median())
                    
                    elif(self._test[i].dtype == 'object'):
                        self._test[i] = self._test[i].replace(na_type, np.NaN)
                        self._test[i] = self._test[i].fillna(dict(self._test[i].mode())[0])

                    else:
                        self._test[i] = self._test[i].replace(na_type, np.NaN)
                        self._test[i] = self._test[i].fillna(self._test[i].mean())

    def DropDuplicates(self):

        if(self._df is not None):
            self._df = self._df.drop_duplicates()
        
        elif(self._train is not None and self._test is not None):
            self._train = self._train.drop_duplicates()
            self._test = self._test.drop_duplicates()
    
    def DropColumn(self, column:str):

        if(self._df is not None):
            self._df = self._df.drop(column, axis=1)

        elif(self._train is not None or self._test is not None):
            self._train = self._train.drop(column, axis=1)
            self._test = self._test.drop(column, axis=1)


    def Encode(self, encode_map:dict):

        label_encoding = LabelEncoder()
        self._ev = [] #Label encoding values

        if(self._df is not None):

            for i in self._df:
                if(self._df[i].dtypes == 'object'):

                    if(i in encode_map.keys()):
                        self._df[i] = self._df[i].replace(encode_map[i])
                        self._ev.append(encode_map[i])
                        continue
                    
                    self._df[i] = label_encoding.fit_transform(self._df[i])
                    self._ev.append(dict(zip(label_encoding.classes_, label_encoding.transform(label_encoding.classes_))))
        
        elif(self._train is not None or self._test is not None):
            for i in self._train:
                if(self._train[i].dtypes == 'object'):
                    self._train[i] = label_encoding.fit_transform(self._train[i])
                    self._ev.append(dict(zip(label_encoding.classes_, label_encoding.transform(label_encoding.classes_))))

            for i in self._test:
                if(self._test[i].dtypes == 'object'):
                    self._test[i] = label_encoding.fit_transform(self._test[i])
                    self._ev.append(dict(zip(label_encoding.classes_, label_encoding.transform(label_encoding.classes_))))
    
    def SetEncodedValues(self, encoded_values:list):
        self._ev = encoded_values.copy()

    def GetEncodedValues(self):
        return self._ev.copy()
        
    def Transform(self, values:list):
        transformed = []
        for i in range(len(values)):
            if(isinstance(values[i], str)):
                added = False
                for col in range(len(self._ev)):
                    if(self._ev[col].get(values[i], -1) != -1):
                        transformed.append(self._ev[col].get(values[i]))
                        added = True
                        break
                
                if(added is False):
                    transformed.append(-1)

            else:
                transformed.append(values[i])
    
        return np.array(transformed)
    
    def GetUnique(self, column_name, get_n = False):

        if self._df is not None:
            return [self._df[column_name].unique(), self._df[column_name].nunique()] if get_n is True else self._df[column_name].unique()
        
        if self._train is not None or self._test is not None:
            raise Exception('Not Complete')
    
    def GetCorrelation(self, annot = True):

        if self._df is not None:

            sns.heatmap(self._df.corr(), annot=annot, cmap='BuPu')
            plt.show()
            return

        elif self._train is not None and self._test is not None:
            temp = pd.concat([self._train, self._test], axis=0)
            sns.heatmap(temp.corr(), annot=annot, cmap='BuPu')
            plt.show()
            return

        print('There is no data')

    def HistPlot(self, column:str):

        if(column is None):
            raise TypeError('No column is selected')

        if(self._df is not None):
            sns.histplot(self._df[column], color='pink')
            plt.show()

        if(self._train is not None or self._test is not None):
            temp = pd.concat([self._train, self._test], axis=0)
            sns.histplot(temp[column], color='pink')
            plt.show()

    def FeatureSplitting(self):
        self._x = self._df.drop(self._target, axis = 1)
        self._y = self._df[self._target]
    
    def TrainTestSplit(self, scale = None, train_size = None, random_state = 0):

        if train_size is None:
            print('Train size is not defined, using 0.8 as the train size')
            train_size = 0.8

        if self._df is not None:

            self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self._x, self._y, train_size=train_size, stratify=self._y, random_state=random_state)

        else:        
            self._x_train = self._train.drop(self._target, axis = 1)
            self._x_test = self._test.drop(self._target, axis = 1)
            self._y_train = self._train[self._target]
            self._y_test = self._test[self._target]
        
        if(scale is not None):
                if scale == 'standard_scaler':
                    scaler = StandardScaler()
                    self._x_train = scaler.fit_transform(self._x_train)
                    self._x_test = scaler.fit_transform(self._x_test)

                elif scale == 'robust_scaler':
                    scaler = RobustScaler()
                    self._x_train = scaler.fit_transform(self._x_train)
                    self._x_test = scaler.fit_transform(self._x_test)
                
                else:
                    raise ValueError('The only supported scaler is \'standard_scaler\' and \'robust_scaler\'')
    
    def TrainTestSplit(self, scale = None, test_size = None, random_state=0):

        if(scale is not None and scale != 'standard_scaler' and scale != 'robust_scaler'):
            raise ValueError('The only supported scaler is \'standard_scaler\' and \'robust_scaler\'')

        if test_size is None:
            print('Test size is not defined, using 0.2 as the test size')
            test_size = 0.2

        if self._df is not None:

            self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self._x, self._y, test_size=test_size, stratify=self._y, random_state=random_state)

        else:        
            self._x_train = self._train.drop(self._target, axis = 1)
            self._x_test = self._test.drop(self._target, axis = 1)
            self._y_train = self._train[self._target]
            self._y_test = self._test[self._target]
        
        if(scale is not None):
                if scale == 'standard_scaler':
                    scaler = StandardScaler()
                    self._x_train = scaler.fit_transform(self._x_train)
                    self._x_test = scaler.fit_transform(self._x_test)

                elif scale == 'robust_scaler':
                    scaler = RobustScaler()
                    self._x_train = scaler.fit_transform(self._x_train)
                    self._x_test = scaler.fit_transform(self._x_test)

    def HandleImbalanceClasses(self, sampling_strategy:float=0.5, task:str='smote'):

        if(sampling_strategy < 0 or sampling_strategy > 1):
            raise ValueError('Sampling strategy can only be 0 to 1 (inclusive)')
        
        if(task != 'random_over_sample' and task != 'random_under_sample' and task != 'smote'):
            raise ValueError('The sampling task can only be \'undersample\' and \'oversample\' and \'smote\'')

        if(task == 'smote'):
            sample = imblearn.over_sampling.SMOTE(sampling_strategy=sampling_strategy, random_state=0)

        if(task == 'random_over_sample'):
            sample = imblearn.over_sampling.RandomOverSampler(sampling_strategy=sampling_strategy, random_state=0)

        if(task == 'random_under_sample'):
            sample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=0)
        
        self._x, self._y = sample.fit_resample(self._x, self._y)

    def ResetData(self):
        self._df = self._original_df.copy() if self._original_df else None
        self._train = self._original_train.copy() if self._original_train else None
        self._test = self._original_test.copy() if self._original_test else None
    
    def GetSplittedData(self):
        return self._x_train, self._x_test, self._y_train, self._y_test
    
    def SetSplittedData(self, splitted_data:list):
        self._x_train, self._x_test, self._y_train, self._y_test = splitted_data