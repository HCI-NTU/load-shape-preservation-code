"""
Implementation of the Deep Temporal Clustering model
Dataset loading functions

@author Florent Forest (FlorentF9)
"""


import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from sklearn.preprocessing import LabelEncoder, Normalizer

ucr = UCR_UEA_datasets()
# UCR/UEA univariate and multivariate datasets.
all_ucr_datasets = ucr.list_datasets()

def moving_avarage_smoothing(X,k):
  	S = np.zeros(X.shape[0])
  	for t in range(X.shape[0]):
    		if t < k:
    			S[t] = np.mean(X[:t+1])
    		else:
    			S[t] = np.sum(X[t-k:t])/k
  	return S

def load_ucr(dataset='CBF'):
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == 'HandMovementDirection':  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert(y.min() == 0)  # assert labels are integers and start from 0
    # preprocess data (standardization)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled, y

def load_energy_use(dataset='housing_size_1_data', time_interval=1, seq_length=96, preprocess='empty', norm='stnd'):
    X_train = np.load('../0_data_2016/' + dataset + '.npy', allow_pickle=True)
    y_train, X_test, y_test = None, None, None
    
    X = []
    for i in range(0, len(X_train)):
        tmp_energy_use = X_train[i]['energy_use'].reshape(seq_length,1).tolist()
        if preprocess == 'cum_sum':
            tmp_energy_use_cum = np.cumsum(tmp_energy_use) # cumulative sum
            tmp_energy_use_cum[0] = 0.000000001 # "loss" converges to 'nan' if the first element is equal to 0
            if time_interval >= 2:
                tmp_energy_use_cum_ = []
                for j in range(0, len(tmp_energy_use_cum)):
                    if j % time_interval == 0:
                        tmp_energy_use_cum_.append(tmp_energy_use_cum[j])
                X.append(tmp_energy_use_cum_)
            else:
                X.append(tmp_energy_use_cum)
            #X.append(tmp_energy_use)
            
        elif preprocess == 'moving_average':
            tmp_energy_use = np.array(tmp_energy_use)
            minpos = np.argmin(tmp_energy_use)
            tmp_energy_use[minpos] = 0.000000001 # "loss" converges to 'nan' if the element is equal to 0
            tmp_energy_use = moving_avarage_smoothing(tmp_energy_use, 4)
            X.append(tmp_energy_use)
            
        else: # preprocess == None
            tmp_energy_use = np.array(tmp_energy_use)
                            
            if time_interval >= 2:
                tmp_energy_use_ = []
                k = 0
                for j in range(0, len(tmp_energy_use)):
                    if j % time_interval == 0:
                        tmp_energy_use_.append(np.sum(tmp_energy_use[k:j]))
                        #print('sum: ', np.sum(tmp_energy_use[k:j]))
                        k = j + 1
                minpos = np.argmin(tmp_energy_use_) 
                tmp_energy_use_[minpos] = 0.00000000001
                X.append(tmp_energy_use_)
            else:
                minpos = np.argmin(tmp_energy_use)
                tmp_energy_use[minpos] = 0.000000001 # "loss" converges to 'nan' if the element is equal to 0
                X.append(tmp_energy_use)
            #if i == 0:
            #    print('tmp_energy_use.shape: ', tmp_energy_use.shape)
            #    print('tmp_energy_use: ', tmp_energy_use)

    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1])
           
    print('X.shape: ', X.shape)
    # preprocess data (min-max normalization, L1, L2)
    #X_scaled = moving_average_smoothing(X)
    if norm == 'stnd':
        X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    elif norm == 'minmax':
        X_scaled = TimeSeriesScalerMinMax(value_range=(0., 1.)).fit_transform(X)
    elif norm == 'l1':
        X = X.reshape(X.shape[0], X.shape[1], 1)
        X_scaled = Normalizer(norm='l1').fit_transform(X) # normalize the data from 0 to 1.
    elif norm == 'none':
        X_scaled = X  

    print('X_scaled.shape: ', X_scaled.shape)
    return X_scaled, y_train

def load_energy_use_whole(seq_length=96, preprocess='empty', norm='stnd'):
    datasets = ['selected_housing_size_1_data', 'selected_housing_size_2_data', 'selected_housing_size_3_data', 'selected_housing_size_4_data', 'selected_housing_size_5_data']
    y_train, X_test, y_test = None, None, None
    
    X_ = []
    for dataset in datasets:
        X_train_ = np.load('../0_data_2020/' + dataset + '.npy', allow_pickle=True)
        #index = np.random.choice(X_train_.shape[0], 10000, replace=False)
        #X_train_ = X_train_[index][:][:]
        
        
        for i in range(0, len(X_train_)):
            energy_use_ = X_train_[i]['energy_use'].reshape(seq_length,1).tolist()
            #energy_use_cum_ = np.cumsum(energy_use_)
            energy_use_cum_ = np.array(energy_use_)
            energy_use_cum_[np.argmin(energy_use_cum_)] = 0.000000001
            X_.append(energy_use_cum_)
     
    X_ = np.array(X_)
    if norm == 'stnd':
        X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_)
    elif norm == 'minmax':
        X_scaled = TimeSeriesScalerMinMax(value_range=(0., 1.)).fit_transform(X_)
    elif norm == 'l1':
        X_ = X_.reshape(X_.shape[0], X_.shape[1], 1)
        X_scaled = Normalizer(norm='l1').fit_transform(X_) # normalize the data from 0 to 1.
    elif norm == 'none':
        X_scaled = X_  
    print('Delete the varaible "X_" and "X_train_"')
    #del(X_)     
    #del(X_train_)
    print('X_scaled.shape: ', X_scaled.shape)
    #print('X_scaled: ', X_scaled)
    return X_scaled, y_train   

def load_data(dataset_name, time_interval, preprocess, norm):
    energy_use_dataset = ['housing_size_1_data', 'housing_size_2_data', 'housing_size_3_data', 'housing_size_4_data', 'housing_size_5_data', 'housing_size_6_data', 'selected_housing_size_1_data', 'selected_housing_size_2_data', 'selected_housing_size_3_data', 'selected_housing_size_4_data', 'selected_housing_size_5_data', 'selected_housing_size_6_data']
    if dataset_name in all_ucr_datasets:
        return load_ucr(dataset_name)
    elif dataset_name in energy_use_dataset:
        return load_energy_use(dataset_name, time_interval=time_interval, preprocess=preprocess, norm=norm)
    elif dataset_name == 'whole':
        return load_energy_use_whole(norm=norm)    
    else:
        print('Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.'.format(dataset_name))
        exit(0)


