import os 
import sys 
import argparse 
 
import cv2 
import numpy as np 

import create_features as cf 
from training import ClassifierTrainer 
 
# Classifying an image 
class ImageClassifier(object): 
    def __init__(self, svm, kmeans, centroids): 
        # Load the SVM classifier 
        self.svm = svm
 
        # Load the codebook 
        self.kmeans, self.centroids = kmeans, centroids
 
    # Method to get the output image tag 
    def getImageTag(self, img): 
        # Resize the input image 
        img = cf.resize_to_size(img) 
 
        # Extract the feature vector 
        feature_vector = cf.FeatureExtractor().get_feature_vector(img, self.kmeans, self.centroids) 
 
        # Classify the feature vector and get the output tag 
        image_tag = self.svm.classify(feature_vector) 
 
        return image_tag 