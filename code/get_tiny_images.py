from PIL import Image 

import pdb
import numpy as np
import math

def get_tiny_images(image_paths):

    '''
    Input : 
        image_paths: a list(N) of string where where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    resize_len = 16;
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path);
        img_resize = img.resize((resize_len,resize_len),Image.BILINEAR);
        img_resize = np.array(img_resize);
        img_resize = np.reshape(img_resize,(1,resize_len*resize_len));
        img_avg = np.average(img_resize);
        img_var = np.var(img_resize);
        img_nor = (img_resize - img_avg) / math.sqrt(img_var);
        if idx == 0:
            tiny_images = img_nor;
        else:
            tiny_images = np.concatenate((tiny_images,img_nor));
        
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
