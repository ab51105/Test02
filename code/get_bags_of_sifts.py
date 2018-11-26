from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import pdb

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
                                                                    
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    
    # You will want to construct SIFT features here in the same way you        #
    # did in build_vocabulary.m (except for possibly changing the sampling     #
    # rate) and then assign each local feature to its nearest cluster center   #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    # load vocab.pkl
    with open('vocab_400.pkl','rb') as f:
        vocab = pickle.load(f)
    centroid_num = vocab.shape[0];
    
    # initial output
    image_feats = np.zeros([len(image_paths),centroid_num])
    
    for idx,path in enumerate(image_paths):
        img = np.asarray(Image.open(path),dtype='float32');
        frames, descriptors = dsift(img, step=[5,5], fast=True);
        dist = distance.cdist(vocab, descriptors, 'euclidean');
        category_result = np.argmin(dist,axis=0);
        hist_value, bins = np.histogram(category_result,bins = range(centroid_num+1));    # range(0,centroid_num)
        normalize = np.linalg.norm(hist_value,ord=1,axis=0);
        if normalize == 0:
            image_feats[idx,:] = hist_value;
        else:
            image_feats[idx,:] = hist_value / normalize;
            
        
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
