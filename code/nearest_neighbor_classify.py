from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    # initialization
    test_predicts = [0 for i in range(test_image_feats.shape[0])];
    
    # calculate the distance
    dist = distance.cdist(test_image_feats,train_image_feats,'euclidean');
       
    # find the nearest-neighbor
    dist_argmin = np.argmin(dist,axis=1);
    for test_num in range(test_image_feats.shape[0]):
        test_predicts[test_num] = train_labels[dist_argmin[test_num]];
        
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
