# -*- coding: utf-8 -*-
"""
@file: machine_learning.py
@author: Matt Chan
@date: Sun Mar 06 14:36:04 2016
@brief: This program loads images of a handwritten characters and uses those images to train
        a classification model. The model can then use that data to identify new characters 
        passed into the model.
@detail: We use supervised learning to train our model (using the Linear Support Vector 
         Classification method) on a certain percentage of the handwritten images. The 
         remainder of the handwritten images are used to test the accuracy of our model.
         For this example, the model is only trained to classify the English characters
         'a', 'b', 'c'.
"""

from scipy.misc import imread
import cv2
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

""" Returns the intervals of pixel rows that contain part of a latter (i.e. black pixels) 
    @param binarized - A matrix of 0s and 1s corresponding to whether the pixel is black or white
    @param axis - Either 0 or 1, corresponding to whether the image is being cropped by columns or rows
"""
def boundaries(binarized,axis):
    # variables named assuming axis = 0; algorithm valid for axis=1
    # [1,0][axis] effectively swaps axes for summing
    rows = np.sum(binarized,axis = [1,0][axis]) > 0
    rows[1:] = np.logical_xor(rows[1:], rows[:-1])
    change = np.nonzero(rows)[0]
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin
    too_small = 10 # real letters will be bigger than 10px by 10px
    ymin = ymin[height>too_small]
    ymax = ymax[height>too_small]
    return zip(ymin,ymax)

""" Crops each letter by removing all columns and rows on the page with whitespace
    @param img - An image containing one column a single English character written multiple times
"""
def separate(img):
    orig_img = img.copy()
    pure_white = 255.
    white = np.max(img)
    black = np.min(img)
    thresh = (white+black)/2.0
    binarized = img<thresh
    row_bounds = boundaries(binarized, axis = 0) 
    cropped = []
    for r1,r2 in row_bounds:
        img = binarized[r1:r2,:]
        col_bounds = boundaries(img,axis=1)
        rects = [r1,r2,col_bounds[0][0],col_bounds[0][1]]
        cropped.append(np.array(orig_img[rects[0]:rects[1],rects[2]:rects[3]]/pure_white))
    return cropped

""" Returns the data and target output for the training set and the test set, given a percentage of the entire data set to train the estimator on 
    @param data_set - The matrix containing all of the images (either 'a', 'b', or 'c')
    @param target_set - An array of target values, ordered to correspond with the data_set
    @param training_ratio - A percentage (between 0 and 1) of the data_set to train our model on. The rest of the data_set will be used to test our model
"""
def partition(data_set, target_set, training_ratio):
    # Number of targets (3 targets here, either 'a', 'b', or 'c')
    n_targets = len(np.unique(target_set))
    print "Number of target values: %d" % n_targets
    # Number of samples taken per target (23 in this example)
    n_samples = len(data)/n_targets
    print "Number of samples per target: %d" % n_samples
    # Size of traning set
    training_samples = int(round(n_samples * training_ratio))
    print "Size of training set: %d" % training_samples
    # Size of test set
    test_samples = int(round(n_samples - training_samples))
    print "Size of test set: %d" % test_samples
    # Array to hold the targets for the training set
    train_target = np.chararray(int(n_targets*training_samples))
    # Array to hold the targes for the testing set
    test_target = np.chararray(int(n_targets*test_samples))
    # Matrix to hold the training data
    train_data = np.empty([len(train_target), len(data[1])])
    # Matrix to hold the test data
    test_data = np.empty([len(test_target), len(data[1])])
    # Initialize values for each array/matrix to its corresponding value
    for target in np.arange(n_targets):
        for i in np.arange(n_samples):
            if i<= training_samples-1:
                train_target[i+(target*training_samples-1)] = target_set[target*n_samples]
                train_data[i+(target*training_samples-1)] = data_set[i+(target*n_samples-1)]
            else:
                test_target[(target*test_samples)+n_samples-i-1] = target_set[target*n_samples]
                test_data[(target*test_samples)+n_samples-i-1] = data_set[i+(target*n_samples-1)]
    return train_data, train_target, test_data, test_target

""" LOAD HANDWRITING IMAGES """
# Read in images, each containing 23 samples of the character 'a', 'b', and 'c'
a_list = imread("a.png", flatten = True) # flatten = True converts to greyscale
b_list = imread("b.png", flatten = True)
c_list = imread("c.png", flatten = True)

numTargets = 3

""" FORMAT IMAGES """
# Crop each list of images around each character
a_imgs = separate(a_list) # separates big_img (pure white = 255) into array of little images (pure white = 1.0)
b_imgs = separate(b_list)
c_imgs = separate(c_list)

data = list()
# Resize images to 10x10 resolution
for img in a_imgs:
    resized_img = cv2.resize(img, (10,10))
    data.append(resized_img)
for img in b_imgs:
    resized_img = cv2.resize(img, (10,10))
    data.append(resized_img)
for img in c_imgs:
    resized_img = cv2.resize(img, (10,10))
    data.append(resized_img)

""" GENERATE DATA SET AND TARGETS """
data = np.asarray(data)
# We now have our data and corresponding targets formatted
data = data.reshape((-1,100))
target = np.chararray(len(data))

# Assign target values to correspond with the data set
for i in np.arange(len(data)):
    if i <=22:
        target[i] = 'a'
    elif 23 <= i <= 45:
        target[i] = 'b'
    elif 46 <= i <= 68:
        target[i] = 'c'
# Split our data set into data to train our estimator on and data to test its accuracy
training_percent = .2
train_data, train_target, test_data, test_target = partition(data,target, training_percent)

""" TRAIN OUR ESTIMATOR AND TEST ITS ACCURACY """
# Use Linear Suppoort Vector Classification to classify the handwriting
clf = svm.LinearSVC()
# Train the estimator on our training set
clf.fit(train_data, train_target)
# Use our estimator to try and predict our testing data set
prediction_set = clf.predict(test_data)
correct = len(prediction_set)
for i in np.arange(len(prediction_set)):
    if (prediction_set[i] != test_target[i]):
        correct -= 1
# Log the accuracy of our estimator
print "Predicted: {0}".format(prediction_set)
print "Truth: {0}".format(test_target)
print "Percent of data used to train model: {0}%".format(training_percent*100)
print "Accuracy: {0}%".format(100*correct/len(prediction_set))

""" VISUALIZE THE OUTPUT """
test_data = test_data.reshape(-1,10,10)
images_and_labels = list(zip(test_data,prediction_set))
# Show some of the predictions along with their corresponding images
for index, (image, label) in enumerate(images_and_labels[0:-1:8]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %s' % label)