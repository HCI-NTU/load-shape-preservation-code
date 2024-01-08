"""
Implementation of the Deep Temporal Clustering model
Main file

@author Florent Forest (FlorentF9)
"""

# Utilities
import os
import csv
import argparse
from time import time
import numpy as np

# Keras
from keras.models import Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2DTranspose, GlobalAveragePooling1D, Softmax
from keras.losses import kullback_leibler_divergence
import keras.backend as K

# TensorFlow
import tensorflow as tf

# scikit-learn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import mean_squared_error, davies_bouldin_score

# Dataset helper function
from datasets import load_data

# DTC components
from TSClusteringLayer import TSClusteringLayer
from TAE import temporal_autoencoder, temporal_autoencoder_v2
from metrics import *
import tsdistances



if __name__ == "__main__":

    # Parsing arguments and setting hyper-parameters
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='selected_housing_size_1_data', help='UCR/UEA univariate or multivariate dataset')
    parser.add_argument('--seq_length', default=4, help='length of one sequence data')
    parser.add_argument('--time_interval', default=24, help='time interval for data sampling (e.g., 24: 6 hours') # this is used only when calculating DBI_time_interval
    parser.add_argument('--preprocess', default=True, type=bool, help='cumulative and time interval sampling? True or False') # this is used only when calculating DBI_time_interval
    parser.add_argument('--clustering_loss', default='kld', help='clustering loss function [kld or dbi]') 
    #parser.add_argument('--validation', default=False, type=bool, help='use train/validation split')
    parser.add_argument('--ae_weights', default=None, help='pre-trained autoencoder weights')
    parser.add_argument('--n_clusters', default=2, type=int, help='number of clusters') # best performance condition: [2, 3, 5, 6, 4]
    parser.add_argument('--n_filters', default=8, type=int, help='number of filters in convolutional layer') # best performance condition: [10, 15, >> 24]
    parser.add_argument('--kernel_size', default=8, type=int, help='size of kernel in convolutional layer') # best performance condition: [8, 4, 12]
    parser.add_argument('--strides', default=1, type=int, help='strides in convolutional layer') 
    parser.add_argument('--pool_size', default=2, type=int, help='pooling size in max pooling layer') # best performance condition: [3, 4, 2]  
    parser.add_argument('--n_units', nargs=2, default=[32, 32, 1], type=int, help='numbers of units in the BiLSTM layers') # best performance condition: [64, 64, -1] [32, 32, -1], [128, 128, -1] [32, -1, -1] [32, 32, 32] [16, 16, -1]
    parser.add_argument('--gamma', default=0.5, type=float, help='coefficient of clustering loss') # best performance condition: [2.0, 1.0, 3.0]
    parser.add_argument('--alpha', default=1.0, type=float, help='coefficient in Student\'s kernel') # best performance condition: [1.0, 2.0]
    parser.add_argument('--dist_metric', default='eucl', type=str, choices=['eucl', 'cid', 'cor', 'acf'], help='distance metric between latent sequences') # best performance condition: cid, cor, eucl
    parser.add_argument('--cluster_init', default='kmeans', type=str, choices=['kmeans', 'hierarchical'], help='cluster initialization method') # 'hierarchical clustering' usually consumes too much memory and computational time. 
    parser.add_argument('--heatmap', default=True, type=bool, help='train heatmap-generating network')
    parser.add_argument('--pretrain_epochs', default=10, type=int)
    parser.add_argument('--epochs', default=30, type=int) # best performance: [60, 100]
    parser.add_argument('--eval_epochs', default=10, type=int)
    parser.add_argument('--save_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=128, type=int) # 128 performed well
    parser.add_argument('--tol', default=0.001, type=float, help='tolerance for stopping criterion')
    parser.add_argument('--patience', default=5, type=int, help='patience for stopping criterion')
    parser.add_argument('--finetune_heatmap_at_epoch', default=8, type=int, help='epoch where heatmap finetuning starts')
    parser.add_argument('--initial_heatmap_loss_weight', default=0.1, type=float, help='initial weight of heatmap loss vs clustering loss')
    parser.add_argument('--final_heatmap_loss_weight', default=0.9, type=float, help='final weight of heatmap loss vs clustering loss (heatmap finetuning)')
    parser.add_argument('--save_dir', default='results/tmp')
    args = parser.parse_args()
    print(args)

    # Create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load data
    (X_train, y_train), (X_val, y_val) = load_data(args.dataset, args.time_interval, args.preprocess), (None, None)  # no train/validation split for now
    #print('X_train: ', X_train)

    # Find number of clusters
    if args.n_clusters is None:
        if y_train == None:
            raise ValueError('To start the clustering without y_train, you need to input "args.n_clusters"')
        args.n_clusters = len(np.unique(y_train))

    # Set default values
    #from keras import optimizers
    #pretrain_optimizer = optimizers.Adam(learning_rate=0.01)
   

    


    # Fit model
    t0 = time()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    cluster = AgglomerativeClustering(n_clusters=args.n_clusters, affinity='euclidean', linkage='complete')  
    y_pred = cluster.fit_predict(X_train)
    print('Training time: ', (time() - t0))
    

    # Evaluate
    print('Performance (TRAIN)')
    results = {}
    
    # Calculate DBI for input data
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    
    print('X_train: ', X_train, ' | X_train.shape: ', X_train.shape)
    print('y_pred: ', y_pred, ' | y_pred.shape: ', y_pred.shape, ' | np.unique(y_pred): ', np.unique(y_pred))
    
#    X_features = dtc.encode(X_train.reshape(X_train.shape[0], X_train.shape[1], 1))
#    X_features = X_features.reshape(X_features.shape[0], X_features.shape[1])
#    print('X_features.shape: ', X_features.shape)
    
    # save X_train, X_features, and y_pred
#    np.save('./results/20201103/X_train_{}_k{}.npy'.format(args.dataset, args.n_clusters), X_train)
#    np.save('./results/20201103/X_features_{}_k{}.npy'.format(args.dataset, args.n_clusters), X_features)
#    np.save('./results/20201103/y_pred_{}_k{}.npy'.format(args.dataset, args.n_clusters), y_pred)
    X_train_raw = np.load('../0_data/X_train_raw_{}.npy'.format(args.dataset))
    DBI = davies_bouldin_score(X_train_raw, y_pred) 
    print("Raw Data DBI for n_clusters = {} is ".format(args.n_clusters), DBI)    
    
    from sklearn.preprocessing import Normalizer, MinMaxScaler
    #X_train_norm = Normalizer(norm='').fit_transform(X_train_raw)
    X_train_norm = MinMaxScaler.fit_transform(X_train_raw)
    DBI = davies_bouldin_score(X_train_norm, y_pred)
    print("Normalized Data DBI for n_clusters = {} is ".format(args.n_clusters), DBI)
        
    # Calculate DBI for 6-hr interval and cum_sum
    DBI = davies_bouldin_score(X_train, y_pred)
    print("6-hr Interval and Cum_Sum DBI for n_clusters = {} is ".format(args.n_clusters), DBI)
    
    # Calculate Silhouette for input data
    sil = silhouette_score(X_train_raw, y_pred)
    print('Silhouette for n_clusters = {} is '.format(args.n_clusters), sil)
    
    sil = silhouette_score(X_train, y_pred)
    print('Latent Feature Silhouette for n_clusters = {} is '.format(args.n_clusters), sil)
    
    """
    # Calculate DBI for a specific time-interval (in this experiment, the interval was set as 6 hour for a fair comparison with Kwonsik's)  
    X_train_interval = []
    for i in range(0, len(X_train)):
        X_train_ = X_train[i]
        X_train_interval_ = []
        for j in range(0, args.seq_length):
            if j % args.time_interval == 1:
                #print('j: ', j)
                X_train_interval_.append(X_train_[j])
        X_train_interval_.append(X_train_[args.seq_length-1])
        X_train_interval.append(X_train_interval_)
    DBI = davies_bouldin_score(X_train_interval, y_pred)
    print("DBI for n_clusters at time_interval = {} and k = {} is ".format(args.time_interval, args.n_clusters), DBI)
    """
