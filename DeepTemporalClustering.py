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
import math
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"

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
from sklearn.neighbors import NearestNeighbors

# Dataset helper function
from datasets import load_data

# DTC components
from TSClusteringLayer import TSClusteringLayer
from TAE import temporal_autoencoder, temporal_autoencoder_v2
from metrics import *
import tsdistances



class DTC:
    """
    Deep Temporal Clustering (DTC) model

    # Arguments
        n_clusters: number of clusters
        input_dim: input dimensionality
        timesteps: length of input sequences (can be None for variable length)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide the time series length
        n_units: numbers of units in the two BiLSTM layers
        alpha: coefficient in Student's kernel
        dist_metric: distance metric between latent sequences
        cluster_init: cluster initialization method

    """

    def __init__(self, n_clusters, input_dim, timesteps,
                 n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1],
                 alpha=1.0, dist_metric='eucl', cluster_init='kmeans', heatmap=False, clustering_loss='custom'):
        assert(timesteps % pool_size == 0)
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.latent_shape = (self.timesteps // self.pool_size, self.n_units[1])
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.cluster_init = cluster_init
        self.heatmap = heatmap
        self.clustering_loss = clustering_loss
        self.pretrained = False
        self.model = self.autoencoder = self.encoder = self.decoder = None
        if self.heatmap:
            self.heatmap_model = None
            self.heatmap_loss_weight = None
            self.initial_heatmap_loss_weight = None
            self.final_heatmap_loss_weight = None
            self.finetune_heatmap_at_epoch = None

    def initialize(self):
        """
        Create DTC model
        """
        # Create AE models
        self.autoencoder, self.encoder, self.decoder = temporal_autoencoder(input_dim=self.input_dim,
        #self.autoencoder, self.encoder, self.decoder = temporal_autoencoder_v2(input_dim=self.input_dim,
                                                                            timesteps=self.timesteps,
                                                                            n_filters=self.n_filters,
                                                                            kernel_size=self.kernel_size,
                                                                            strides=self.strides,
                                                                            pool_size=self.pool_size,
                                                                            n_units=self.n_units)
        clustering_layer = TSClusteringLayer(self.n_clusters,
                                             alpha=self.alpha,
                                             dist_metric=self.dist_metric,
                                             name='TSClustering')(self.encoder.output)

        # Heatmap-generating network
        if self.heatmap:
            n_heatmap_filters = self.n_clusters  # one heatmap (class activation map) per cluster
            encoded = self.encoder.output
            heatmap_layer = Reshape((-1, 1, self.n_units[1]))(encoded)
            heatmap_layer = UpSampling2D((self.pool_size, 1))(heatmap_layer)
            heatmap_layer = Conv2DTranspose(n_heatmap_filters, (self.kernel_size, 1), padding='same')(heatmap_layer)
            # The next one is the heatmap layer we will visualize
            heatmap_layer = Reshape((-1, n_heatmap_filters), name='Heatmap')(heatmap_layer)
            heatmap_output_layer = GlobalAveragePooling1D()(heatmap_layer)
            # A dense layer must be added only if `n_heatmap_filters` is different from `n_clusters`
            # heatmap_output_layer = Dense(self.n_clusters, activation='relu')(heatmap_output_layer)
            heatmap_output_layer = Softmax()(heatmap_output_layer)  # normalize activations with softmax

        if self.heatmap:
            # Create DTC model
            self.model = Model(inputs=self.autoencoder.input,
                               outputs=[self.autoencoder.output, clustering_layer, heatmap_output_layer])
            # Create Heatmap model
            self.heatmap_model = Model(inputs=self.autoencoder.input,
                                       outputs=heatmap_layer)
        elif self.clustering_loss == 'custom':
            # Create DTC model
            print("We only trained DTC model: encoder, clsutering layer, and locality preserving")
            
            """---------------------------------------------y_pred----------------------------------------------------"""
            """---------------------------------------------y_pred----------------------------------------------------"""
            """---------------------------------------------y_pred----------------------------------------------------"""
            self.model = Model(inputs=self.autoencoder.input,
                               outputs=[self.autoencoder.output, clustering_layer, self.encoder.output])
        else:
            self.model = Model(inputs=self.autoencoder.input,
                               outputs=[self.autoencoder.output, clustering_layer])        

    @property
    def cluster_centers_(self):
        """
        Returns cluster centers
        """
        return self.model.get_layer(name='TSClustering').get_weights()[0]
#['mse', DTC.weighted_kld(1.0 - self.heatmap_loss_weight), DTC.weighted_kld(self.heatmap_loss_weight)]
    @staticmethod
    def weighted_kld(loss_weight):
        """
        Custom KL-divergence loss with a variable weight parameter
        """
        def loss(y_true, y_pred):
            return loss_weight * kullback_leibler_divergence(y_true, y_pred)
        return loss
    
    def inverse_weight_distance(self, distances):
        """
        num_of_data = distances.shape[0]
    
        for i in range(0, num_of_data):
            distance = distances[i]
            distance = distance[1:] # remove the zero elements, which are the distances between the same data points.
    
            weight = 1.0 / distance
            weight = weight / np.sum(weight) # make the sum of weight to be 1
    
            if i == 0:
                weights = weight
            else:
                weights = np.append(weights, weight)
    
        return weights.reshape(distances.shape[0], distances.shape[1]-1)
        """
        #weight = 1.0 / distances
        #print('distance in the inverse: ', K.eval(distances))
        weight = tf.math.divide(1.0, distances, name=None)
        weight = self.replace_inf_with_zero(weight)
        weight = self.replace_nan_with_zero(weight)
        #print('weight in the inverse: ', K.eval(weight))
        #print('K.sum(weight).shape', K.sum(weight, axis=1).shape)
        #tf.expand_dims(image, axis=0)
        return weight / tf.expand_dims(K.sum(weight, axis=1), axis=-1)
 
        
    def replace_inf_with_zero(self, X):
        return tf.where(tf.is_inf(X), tf.zeros_like(X), X)   
    
    def replace_nan_with_zero(self, X):
        #index_nan = np.isnan(X)
        #X[index_nan] = 0
        return tf.where(tf.is_nan(X), tf.zeros_like(X), X)   
        #return X   
    
    #@staticmethod
    #def locality_preserving_loss(self, X_train,  X_features, n_neighbors=5):
    def mean(labels):
        return sum(labels) / len(labels)
    
    def euclidean_distance(point1, point2):
        sum_squared_distance = 0
        for i in range(len(point1)):
            sum_squared_distance += math.pow(point1[i] - point2[i], 2)
        return math.sqrt(sum_squared_distance)
        
    def locality_preserving_loss(self, n_neighbors=5):
        """    
        #This loss is to ensure that similar datapoints in the original space should be located near enough with each other in the feature space as well
        
        #X: datapoints in the original space
        #X_features: datapoints in the feature space (encoded data)
        #n_neighbors: the number of neighbors of each datapoint
        """
    
        def loss(y_true, y_pred):
            # Step 1: find the k-nearest neighbors, calculate their distances in the original space, and compute the weight valuese
            # (that allow to assign more importance on the nearest data points)
            #sess = K.get_session()
            #array = sess.run(your_tensor)
            #X = sess.run(y_true)
            #X = y_true.eval(session=tf.compat.v1.Session())      
            #X = y_true.eval(session=sess)   
            X = K.constant(y_true)
            #X_features = y_pred.eval(session=K.get_session())
            print('y_true.shape: ', y_true)
            
            """
            distances = K.sqrt(K.sum(K.square(X - X_features), axis=-1))
            distances = tf.sort(distances)
            print(distances)
            indices = tf.argsort(distances,axis=-1,direction='ASCENDING',stable=False,name=None)
            """
                     
            neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(X.numpy())
            distances, indices = neighbors.kneighbors(X) 
            
            weights = self.inverse_weight_distance(distances)
            weights = self.replace_nan_with_zero(weights) # remove the "NaN" elements. 
            print('*******************************************************************************************************************')
                        
            # Step 2: calculate the distances among neighbors in the feature space
            #X_features = DTC.encode(self, X)
            X_features_neighbors = X_features[indices] # shape: (26645, n_neighbors, 96)
            for i in range(0, X_features_neighbors.shape[0]):
                X_features_neighbors_ = X_features_neighbors[i]
                X_features_centroid = X_features_neighbors_[0]
                
                feature_distance = np.linalg.norm(X_features_neighbors_[1:] - X_features_centroid, axis=1)
                feature_distance = feature_distance.reshape(1,feature_distance.shape[0])
                
                if i == 0:
                    feature_distances = feature_distance
                else:
                    feature_distances = np.append(feature_distances, feature_distance, axis=0)
                    
            # Step 3: multiply the weight and the feature distances
            locality_preserving_loss = weights * feature_distances
            locality_preserving_loss = np.sum(locality_preserving_loss, axis=1)
            locality_preserving_loss = locality_preserving_loss.mean()
            
            return locality_preserving_loss
        return loss
    
    def pairwise_dist(self, A, B):  
        """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
          A,    [m,d] matrix
          B,    [n,d] matrix
        Returns:
          D,    [m,n] matrix of pairwise distances
        """
        
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)
        
        # na as a row and nb as a column vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
    
        # return pairwise euclidead difference matrix
        D=tf.reduce_sum((tf.expand_dims(A, 1)-tf.expand_dims(B, 0))**2,2)
        return D
        
    def getitems_by_indices(self, values, indices):
        return tf.map_fn(
            lambda x: tf.gather(x[0], x[1]), (values, indices), dtype=values.dtype
        )
        
    def locality_preserving_loss_K(self, n_neighbors=5, batch_size=128, seq_length=96, feature_length=48, threshold_distance=0.5):
        """    
        #This loss is to ensure that similar datapoints in the original space should be located near enough with each other in the feature space as well
        
        #X: datapoints in the original space
        #X_features: datapoints in the feature space (encoded data)
        #n_neighbors: the number of neighbors of each datapoint
        """
    
        def loss(y_true, y_pred):
            # y_true: (batch_size, 96, 1) <- it is defined in the function "fit"
            # y_pred: (batch_size, 48, 1) <- it is defined in the function "initialize"
            #print('y_true.shape: ', y_true.shape)
            #print('y_pred.shape: ', y_pred.shape)
            X_train = y_true
            X_train = tf.reshape(X_train, [batch_size, seq_length]) 
            #X_feature = self.encode(X_train)
            X_feature = y_pred
            X_feature = tf.reshape(X_feature, [batch_size, feature_length]) 
                        
            # Step 1: calculate distances between data points (batch_size) in "X_train" to find the k-nearest neighbors
            X_train_distance = self.pairwise_dist(X_train, X_train) # calculate distances within "X_train"
            #print('---------------------------------------X_train_distance.shape: ', X_train_distance.shape)
            X_train_distance_neighbors, index_neighbors = tf.nn.top_k(tf.negative(X_train_distance), k=n_neighbors, sorted=True) # whether the variable "sorted" is set as True or False, the results are not different -> so we need to make X_train_distance as negative values to get the nearest k points
            X_train_distance_neighbors = tf.math.abs(X_train_distance_neighbors) # make the distance as the positive again
            
            # replace the distances greater than the threshold_distance (top_knn) with zeros to ignore the too-far cases <- weights will be calculated as 0.
            X_train_distance_neighbors = tf.where(X_train_distance_neighbors < threshold_distance, X_train_distance_neighbors, tf.zeros_like(X_train_distance_neighbors)) 
            
            weights = self.inverse_weight_distance(X_train_distance_neighbors)
            weights = self.replace_nan_with_zero(weights)
            #print('K.eval(weights): ', K.eval(weights))
            
            # Step 2: calculate the distances among neighbors in the feature space 
            # calcualte the distances in the feature space
            X_feature_distance = self.pairwise_dist(X_feature, X_feature)
            print('------------------------X_feature_distance.shape: ', X_feature_distance.shape)
            #index_neighbors = tf.stack([tf.range(index_neighbors.shape[0])[:, n_neighbors], index_neighbors], axis=2)
            
            # prepare the indices array before getting the feature distances
            basepoints = tf.range(batch_size)[:,None]
            for i in range(1, n_neighbors):
                basepoints_ = tf.range(batch_size)[:,None]
                basepoints = tf.concat([basepoints, basepoints_], axis=1)
            index_neighbors = tf.reshape(index_neighbors, [batch_size,n_neighbors,1])
            index_neighbors = tf.concat([basepoints[:,:,None], index_neighbors], axis=2)
            
            # finally get the distances of neighboring datapoints from basepoints in the feature space
            X_feature_distance_neighbors = tf.gather_nd(X_feature_distance, index_neighbors)
    
            
            # Step 3: multiply the weight and the feature distances
            locality_preserving_loss = tf.math.multiply(weights, X_feature_distance_neighbors, name=None) # multiply the weights (that are calculated based on the distance in the original space) and the distances in the feature space
            locality_preserving_loss = K.sum(locality_preserving_loss, axis=1)
            locality_preserving_loss = tf.math.reduce_mean(locality_preserving_loss)
            
            return locality_preserving_loss
        return loss
    
    def on_epoch_end(self, epoch):
        """
        Update heatmap loss weight on epoch end
        """
        if epoch > self.finetune_heatmap_at_epoch:
            K.set_value(self.heatmap_loss_weight, self.final_heatmap_loss_weight)
    
    @staticmethod
    def MSE_DBI_loss(X_train):
        features = self.encode(X_train)
        
        loss_mse = mean_squared_error(X_train, features)
        
        #hc = AgglomerativeClustering(n_clusters=self.n_clusters,
        #                                     affinity='euclidean',
        #                                     linkage='complete').fit(features.reshape(features.shape[0], -1))
        km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(features.reshape(features.shape[0], -1))
        
        loss_dbi = (features.reshape(X_train.shape[0], X_train.shape[1]), hc)
        print("DBI_loss: features.shape: ", X_train.shape)
        print("DBI_loss: hc.shape: ", hc.shape)
        
        return loss_mse + loss_dbi
    
    def compile(self, X_train, gamma, optimizer, clustering_loss='kld', loss_weights=[1.0, 1.0, 1.0], n_neighbors=5, batch_size=128, feature_length=48, threshold_distance=0.5, initial_heatmap_loss_weight=None, final_heatmap_loss_weight=None):
        """
        Compile DTC model

        # Arguments
            gamma: coefficient of TS clustering loss
            optimizer: optimization algorithm
            initial_heatmap_loss_weight (optional): initial weight of heatmap loss vs clustering loss
            final_heatmap_loss_weight (optional): final weight of heatmap loss vs clustering loss (heatmap finetuning)
        """
        if self.heatmap:
            self.initial_heatmap_loss_weight = initial_heatmap_loss_weight
            self.final_heatmap_loss_weight = final_heatmap_loss_weight
            self.heatmap_loss_weight = K.variable(self.initial_heatmap_loss_weight)
            self.model.compile(loss=['mse', DTC.weighted_kld(1.0 - self.heatmap_loss_weight), DTC.weighted_kld(self.heatmap_loss_weight)],
                               loss_weights=[1.0, gamma, gamma],
                               optimizer=optimizer)
        elif clustering_loss == 'kld':
            self.model.compile(loss=['mse', 'kld'],
                               loss_weights=[1.0, gamma],
                               optimizer=optimizer)
        elif clustering_loss == 'dbi':
             self.model.compile(loss=self.MSE_DBI_loss(X_train),
                               optimizer=optimizer)
        elif clustering_loss == 'custom': # reconstruction, clustering, locality-preserving
            print('')
            print('======================n_neighbors for locality-preserving_loss: ', n_neighbors)
            print('')
            self.model.compile(loss=['mse', 'kld', DTC.locality_preserving_loss_K(self, batch_size=batch_size, n_neighbors=n_neighbors, feature_length=feature_length, threshold_distance=threshold_distance)],
                               loss_weights=loss_weights,
                               optimizer=optimizer)                                       
        
    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DTC model

        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """
        Load pre-trained weights of AE

        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def dist(self, x1, x2):
        """
        Compute distance between two multivariate time series using chosen distance metric

        # Arguments
            x1: first input (np array)
            x2: second input (np array)
        # Return
            distance
        """
        if self.dist_metric == 'eucl':
            return tsdistances.eucl(x1, x2)
        elif self.dist_metric == 'cid':
            return tsdistances.cid(x1, x2)
        elif self.dist_metric == 'cor':
            return tsdistances.cor(x1, x2)
        elif self.dist_metric == 'acf':
            return tsdistances.acf(x1, x2)
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')

    def init_cluster_weights(self, X):
        """
        Initialize with complete-linkage hierarchical clustering or k-means.

        # Arguments
            X: numpy array containing training set or batch
        """
        assert(self.cluster_init in ['hierarchical', 'kmeans'])
        print('Initializing cluster...')

        features = self.encode(X)
        print('features.shape: ', features.shape)

        if self.cluster_init == 'hierarchical':
            if self.dist_metric == 'eucl':  # use AgglomerativeClustering off-the-shelf
                hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity='euclidean',
                                             linkage='complete').fit(features.reshape(features.shape[0], -1))
            else:  # compute distance matrix using dist
                d = np.zeros((features.shape[0], features.shape[0]))
                for i in range(features.shape[0]):
                    for j in range(i):
                        d[i, j] = d[j, i] = self.dist(features[i], features[j])
                hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity='precomputed',
                                             linkage='complete').fit(d)
            # compute centroid
            cluster_centers = np.array([features[hc.labels_ == c].mean(axis=0) for c in range(self.n_clusters)])
        elif self.cluster_init == 'kmeans':
            # fit k-means on flattened features
            km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(features.reshape(features.shape[0], -1))
            cluster_centers = km.cluster_centers_.reshape(self.n_clusters, features.shape[1], features.shape[2])

        self.model.get_layer(name='TSClustering').set_weights([cluster_centers])
        print('Done!')

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer

        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)

    def decode(self, x):
        """
        Decoding function. Decodes encoded sequence from latent space.

        # Arguments
            x: encoded (latent) data point
        # Return
            decoded data point
        """
        return self.decoder.predict(x)

    def predict(self, x):
        """
        Predict cluster assignment.

        """
        q = self.model.predict(x, verbose=0)[1]
        return q.argmax(axis=1)

    @staticmethod
    def target_distribution(q):  # target distribution p which enhances the discrimination of soft label q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def predict_heatmap(self, x):
        """
        Produces TS clustering heatmap from input sequence.

        # Arguments
            x: data point
        # Return
            heatmap
        """
        return self.heatmap_model.predict(x, verbose=0)

    def pretrain(self, X,
                 optimizer='adam',
                 epochs=10,
                 batch_size=64,
                 save_dir='results/tmp',
                 verbose=1):
        """
        Pre-train the autoencoder using only MSE reconstruction loss
        Saves weights in h5 format.

        # Arguments
            X: training set
            optimizer: optimization algorithm
            epochs: number of pre-training epochs
            batch_size: training batch size
            save_dir: path to existing directory where weights will be saved
        """
        print('Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse') ################ need to add "locality_preserving_loss" here to maintain shape information?
        
        # Begin pretraining
        t0 = time()
        self.autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, verbose=verbose)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights('{}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        print('Pretrained weights are saved to {}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        self.pretrained = True

    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            epochs=10,
            eval_epochs=10,
            save_epochs=10,
            batch_size=64,
            tol=0.001,
            patience=5,
            finetune_heatmap_at_epoch=8,
            save_dir='results/tmp',
            clustering_loss='kld'):
        """
        Training procedure

        # Arguments
           X_train: training set
           y_train: (optional) training labels
           X_val: (optional) validation set
           y_val: (optional) validation labels
           epochs: number of training epochs
           eval_epochs: evaluate metrics on train/val set every eval_epochs epochs
           save_epochs: save model weights every save_epochs epochs
           batch_size: training batch size
           tol: tolerance for stopping criterion
           patience: patience for stopping criterion
           finetune_heatmap_at_epoch: epoch number where heatmap finetuning will start. Heatmap loss weight will
                                      switch from `self.initial_heatmap_loss_weight` to `self.final_heatmap_loss_weight`
           save_dir: path to existing directory where weights and logs are saved
        """
        if not self.pretrained:
            print('Autoencoder was not pre-trained!')

        if self.heatmap:
            self.finetune_heatmap_at_epoch = finetune_heatmap_at_epoch

        # Logging file
        logfile = open(save_dir + '/dtc_log.csv', 'w')
        fieldnames = ['epoch', 'T', 'L', 'Lr', 'Lc', 'Ll']
        if X_val is not None:
            fieldnames += ['L_val', 'Lr_val', 'Lc_val']
        if y_train is not None:
            fieldnames += ['acc', 'pur', 'nmi', 'ari']
        if y_val is not None:
            fieldnames += ['acc_val', 'pur_val', 'nmi_val', 'ari_val']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        y_pred_last = None
        patience_cnt = 0

        print('Training for {} epochs.\nEvaluating every {} and saving model every {} epochs.'.format(epochs, eval_epochs, save_epochs))

        for epoch in range(epochs):

            # Compute cluster assignments for training set
            q = self.model.predict(X_train)[1]
            p = DTC.target_distribution(q)

            #dbi = DTC.DBI_loss(X_train)

            # Evaluate losses and metrics on training set
            if epoch % eval_epochs == 0:

                # Initialize log dictionary
                logdict = dict(epoch=epoch)

                y_pred = q.argmax(axis=1)
                if X_val is not None:
                    q_val = self.model.predict(X_val)[1]
                    p_val = DTC.target_distribution(q_val)
                    y_val_pred = q_val.argmax(axis=1)

                print('epoch {}'.format(epoch))
                if self.heatmap:
                    loss = self.model.evaluate(X_train, [X_train, p, p], batch_size=batch_size, verbose=False)
                elif clustering_loss == 'custom':
                    """---------------------------evaluate----------------------------"""
                    #X_feature = self.encode(X_train)
                    #print('X_feature.shape: ', X_feature.shape)
                    #print('p.shape: ', p.shape)
                    #print('X_train.shape: ', X_train.shape)
                    loss = self.model.evaluate(X_train, [X_train, p, X_train], batch_size=batch_size, verbose=False)
                    #loss = self.model.evaluate(X_train, [X_train, dbi], batch_size=batch_size, verbose=False)
                else:
                    loss = self.model.evaluate(X_train, [X_train, p], batch_size=batch_size, verbose=False)
                logdict['L'] = loss[0]
                logdict['Lr'] = loss[1]
                logdict['Lc'] = loss[2]
                if clustering_loss == 'custom':
                    logdict['Ll'] = loss[3]
                else:
                    logdict['Ll'] = 0
                print('[Train] - Lr={:f}, Lc={:f}, Ll={:f} - total loss={:f}'.format(logdict['Lr'], logdict['Lc'], logdict['Ll'], logdict['L']))
                if X_val is not None:
                    val_loss = self.model.evaluate(X_val, [X_val, p_val], batch_size=batch_size, verbose=False)
                    logdict['L_val'] = val_loss[0]
                    logdict['Lr_val'] = val_loss[1]
                    logdict['Lc_val'] = val_loss[2]
                    logdict['Ll_val'] = val_loss[3]
                    print('[Val] - Lr={:f}, Lc={:f} - total loss={:f}'.format(logdict['Lr_val'], logdict['Lc_val'], logdict['L_val']))

                # Evaluate the clustering performance using labels
                if y_train is not None:
                    logdict['acc'] = cluster_acc(y_train, y_pred)
                    logdict['pur'] = cluster_purity(y_train, y_pred)
                    logdict['nmi'] = metrics.normalized_mutual_info_score(y_train, y_pred)
                    logdict['ari'] = metrics.adjusted_rand_score(y_train, y_pred)
                    print('[Train] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc'], logdict['pur'],
                                                                                    logdict['nmi'], logdict['ari']))
                if y_val is not None:
                    logdict['acc_val'] = cluster_acc(y_val, y_val_pred)
                    logdict['pur_val'] = cluster_purity(y_val, y_val_pred)
                    logdict['nmi_val'] = metrics.normalized_mutual_info_score(y_val, y_val_pred)
                    logdict['ari_val'] = metrics.adjusted_rand_score(y_val, y_val_pred)
                    print('[Val] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc_val'], logdict['pur_val'],
                                                                                  logdict['nmi_val'], logdict['ari_val']))

                logwriter.writerow(logdict)

                # check stop criterion
                if y_pred_last is not None:
                    assignment_changes = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if epoch > 0 and assignment_changes < tol:
                    patience_cnt += 1
                    print('Assignment changes {} < {} tolerance threshold. Patience: {}/{}.'.format(assignment_changes, tol, patience_cnt, patience))
                    if patience_cnt >= patience:
                        print('Reached max patience. Stopping training.')
                        logfile.close()
                        break
                else:
                    patience_cnt = 0

            # Save intermediate model and plots
            if epoch % save_epochs == 0:
                self.model.save_weights(save_dir + '/DTC_model_' + str(epoch) + '.h5')
                print('Saved model to:', save_dir + '/DTC_model_' + str(epoch) + '.h5')

            # Train for one epoch
            if self.heatmap:
                self.model.fit(X_train, [X_train, p, p], epochs=1, batch_size=batch_size, verbose=False)
                self.on_epoch_end(epoch)
            elif clustering_loss == 'custom':
                self.model.fit(X_train, [X_train, p, X_train], epochs=1, batch_size=batch_size, verbose=False)
            else:
                self.model.fit(X_train, [X_train, p], epochs=1, batch_size=batch_size, verbose=False)
                """---------------------------fit----------------------------"""
                
                #self.model.fit(X_train, [X_train, dbi], epochs=1, batch_size=batch_size, verbose=False)

        # Save the final model
        logfile.close()
        print('Saving model to:', save_dir + '/DTC_model_final.h5')
        self.model.save_weights(save_dir + '/DTC_model_final.h5')


if __name__ == "__main__":
    # Parsing arguments and setting hyper-parameters
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='whole', help='UCR/UEA univariate or multivariate dataset')
    parser.add_argument('--housing_size', default='all')
    parser.add_argument('--seq_length', default=96, help='length of one sequence data')
    parser.add_argument('--time_interval', default=1, help='time interval for data sampling (e.g., 24: 6 hours') # this is used only when calculating DBI_time_intervald
    parser.add_argument('--preprocess', default='others', help='"cum_sum" or "moving_average" or "others(standardization, it can be changed in another script "datasets")"')
    parser.add_argument('--norm', default='stnd', help='types of normalization: "stnd" "l1" "l2" "minmax" and "none"')
    parser.add_argument('--clustering_loss', default='kld', help='clustering loss function [kld or custom] -> custom: reconstruction, kld, and locality-preserving')
    parser.add_argument('--loss_weights', default=[1.0, 1.0, 1.0], help='weights among losses: [reconstruction, clustering, locality-preserving]')
    parser.add_argument('--n_neighbors', default=30, help='numbers of neighbors when calculating the locality-preserving loss') 
    #parser.add_argument('--validation', default=False, type=bool, help='use train/validation split')
    parser.add_argument('--ae_weights', default=None, help='pre-trained autoencoder weights')
    parser.add_argument('--n_clusters', default=10, type=int, help='number of clusters') # best performance condition: [2, 3, 5, 6, 4]
    parser.add_argument('--n_filters', default=4, type=int, help='number of filters in convolutional layer') # best performance condition: [10, 15, >> 24]
    parser.add_argument('--kernel_size', default=4, type=int, help='size of kernel in convolutional layer') # best performance condition: [8, 4, 12]
    parser.add_argument('--strides', default=1, type=int, help='strides in convolutional layer') 
    parser.add_argument('--pool_size', default=3, type=int, help='pooling size in max pooling layer') # best performance condition: [3, 4, 2]  
    parser.add_argument('--n_units', default=[32, 32, 1], type=int, help='numbers of units in the BiLSTM layers') # best performance condition: [64, 64, -1] [32, 32, -1], [128, 128, -1] [32, -1, -1] [32, 32, 32] [16, 16, -1]

    parser.add_argument('--gamma', default=1.0, type=float, help='coefficient of clustering loss') # 
    parser.add_argument('--alpha', default=1.0, type=float, help='coefficient in Student\'s kernel') #
    parser.add_argument('--dist_metric', default='eucl', type=str, choices=['eucl', 'cid', 'cor', 'acf'], help='distance metric between latent sequences') # best performance condition: cid, cor, eucl
    parser.add_argument('--cluster_init', default='kmeans', type=str, choices=['kmeans', 'hierarchical'], help='cluster initialization method') # 'hierarchical clustering' usually consumes too much memory and computational time. 
    parser.add_argument('--heatmap', default=False, type=bool, help='train heatmap-generating network')
    # if heatmap is set as "True", we need to modifty the order of model outputs in the "initialize" function in this script.
    parser.add_argument('--pretrain_epochs', default=10, type=int) # The value of 5 allows to balance btw reconstruction and clustering losses
    parser.add_argument('--epochs', default=30, type=int) # The value of 30 allows to balance btw reconstruction and clustering losses
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--save_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=128, type=int) # 128 performed well
    parser.add_argument('--tol', default=0.001, type=float, help='tolerance for stopping criterion')
    parser.add_argument('--patience', default=5, type=int, help='patience for stopping criterion')
    parser.add_argument('--finetune_heatmap_at_epoch', default=8, type=int, help='epoch where heatmap finetuning starts')
    parser.add_argument('--initial_heatmap_loss_weight', default=0.1, type=float, help='initial weight of heatmap loss vs clustering loss')
    parser.add_argument('--final_heatmap_loss_weight', default=0.9, type=float, help='final weight of heatmap loss vs clustering loss (heatmap finetuning)')
    parser.add_argument('--save_dir', default='results/data2020')
    
    
    args = parser.parse_args()
    feature_length = int(args.seq_length / args.pool_size)
    args.save_dir = args.save_dir + '_epoch{}_HS{}_seq{}_k{}_neighbor{}_batchsize{}_lossweight{}'.format(args.epochs, args.housing_size, feature_length,        args.n_clusters, args.n_neighbors, args.batch_size, int(args.loss_weights[2]))
    print(args)

    # Create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load data
    (X_train, y_train), (X_val, y_val) = load_data(args.dataset, args.time_interval, args.preprocess, args.norm), (None, None)  # no train/validation split for now
    #print('X_train: ', X_train)
    
    # Calculate the maximum distance between neighbors (max(disance(top_knn))): "threshold_distance"
    """
    neighbors = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm='ball_tree', metric='euclidean').fit(X_train.reshape(X_train.shape[0], X_train.shape[1]))
    distances, indices = neighbors.kneighbors(X_train.reshape(X_train.shape[0], X_train.shape[1]))
    threshold_distance = np.amax(distances) # the maximum distance among all parwise distances
    """
    threshold_distance = 11.4441

    
    # Find number of clusters
    if args.n_clusters is None:
        if y_train == None:
            raise ValueError('To start the clustering without y_train, you need to input "args.n_clusters"')
        args.n_clusters = len(np.unique(y_train))

    # Set default values
    #from keras import optimizers
    #pretrain_optimizer = optimizers.Adam(learning_rate=0.01)
    pretrain_optimizer = 'adam'
    
    # Instantiate model
    dtc = DTC(n_clusters=args.n_clusters,
              input_dim=X_train.shape[-1],
              timesteps=X_train.shape[1],
              n_filters=args.n_filters,
              kernel_size=args.kernel_size,
              strides=args.strides,
              pool_size=args.pool_size,
              n_units=args.n_units,
              alpha=args.alpha,
              dist_metric=args.dist_metric,
              cluster_init=args.cluster_init,
              heatmap=args.heatmap,
              clustering_loss=args.clustering_loss)

    # Initialize model
    optimizer = 'adam'
    dtc.initialize()
    dtc.model.summary()
    dtc.compile(X_train=X_train, gamma=args.gamma, optimizer=optimizer, clustering_loss=args.clustering_loss, loss_weights=args.loss_weights, n_neighbors=args.n_neighbors, batch_size=args.batch_size, feature_length=feature_length, threshold_distance=threshold_distance, initial_heatmap_loss_weight=args.initial_heatmap_loss_weight, final_heatmap_loss_weight=args.final_heatmap_loss_weight)

    # Load pre-trained AE weights or pre-train
    if args.ae_weights is None and args.pretrain_epochs > 0:
        dtc.pretrain(X=X_train, optimizer=pretrain_optimizer,
                     epochs=args.pretrain_epochs, batch_size=args.batch_size,
                     save_dir=args.save_dir)
    elif args.ae_weights is not None:
        dtc.load_ae_weights(args.ae_weights)
    
    # Initialize clusters
    dtc.init_cluster_weights(X_train)

    # Fit model
    t0 = time()
    dtc.fit(X_train[:args.batch_size*int(X_train.shape[0]/args.batch_size),:,:], y_train, X_val, y_val, args.epochs, args.eval_epochs, args.save_epochs, args.batch_size,
            args.tol, args.patience, args.finetune_heatmap_at_epoch, args.save_dir, args.clustering_loss)
    print('Training time: ', (time() - t0))
    

    # Evaluate
    print('Performance (TRAIN)')
    results = {}
    q = dtc.model.predict(X_train)[1]
    y_pred = q.argmax(axis=1)
    if y_train is not None:
        results['acc'] = cluster_acc(y_train, y_pred)
        results['pur'] = cluster_purity(y_train, y_pred)
        results['nmi'] = metrics.normalized_mutual_info_score(y_train, y_pred)
        results['ari'] = metrics.adjusted_rand_score(y_train, y_pred)
    print(results)
    
    # Calculate DBI for input data
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    #X_train_raw = np.load('../0_data/X_train_raw_{}.npy'.format(args.dataset))
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    print('X_train: ', X_train, ' | X_train.shape: ', X_train.shape)
    print('y_pred: ', y_pred, ' | y_pred.shape: ', y_pred.shape, ' | np.unique(y_pred): ', np.unique(y_pred))
    
    X_features = dtc.encode(X_train.reshape(X_train.shape[0], X_train.shape[1], 1))
    X_features = X_features.reshape(X_features.shape[0], X_features.shape[1])
    print('X_features.shape: ', X_features.shape)
    
    # save X_train, X_features, and y_pred
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    np.save('{}/X_train_{}_k{}.npy'.format(args.save_dir, args.dataset, args.n_clusters), X_train)
    np.save('{}/X_features_{}_k{}.npy'.format(args.save_dir, args.dataset, args.n_clusters), X_features)
    np.save('{}/y_pred_{}_k{}.npy'.format(args.save_dir, args.dataset, args.n_clusters), y_pred)
    
    #DBI = davies_bouldin_score(X_train_raw, y_pred) 
    #print("Raw Data DBI for n_clusters = {} is ".format(args.n_clusters), DBI) # When calculating DBI, I think I need to change it to the same time-interval with Kwonsik's. Maybe the number of time-steps would be different from Kwonsik's (maybe, 6 hour time-step)    

    DBI = davies_bouldin_score(X_train, y_pred) 
    print("Normalized Data DBI for n_clusters = {} is ".format(args.n_clusters), DBI) # When calculating DBI, I think I need to change it to the same time-interval with Kwonsik's. Maybe the    
    # Calculate DBI for latent features

    DBI = davies_bouldin_score(X_features, y_pred)
    print("Latent Feature DBI for n_clusters = {} is ".format(args.n_clusters), DBI)
    
    # Calculate Silhouette for input data
    #sil = silhouette_score(X_train_raw, y_pred)
    #print('Silhouette for n_clusters = {} is '.format(args.n_clusters), sil)
    
    #sil = silhouette_score(X_features, y_pred)
    #print('Latent Feature Silhouette for n_clusters = {} is '.format(args.n_clusters), sil)
    
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
