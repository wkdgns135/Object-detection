import os 
import sys 
import argparse 
 
import _pickle as pickle
import numpy as np 
from sklearn.multiclass import OneVsOneClassifier 
from sklearn.svm import LinearSVC 
from sklearn import preprocessing 
 
# To train the classifier 
class ClassifierTrainer(object): 
    def __init__(self, X, label_words): 
        # Encoding the labels (words to numbers) 
        self.le = preprocessing.LabelEncoder() 
 
        # Initialize One vs One Classifier using a linear kernel 
        self.clf = OneVsOneClassifier(LinearSVC(random_state=0)) 
 
        y = self._encodeLabels(label_words) 
        X = np.asarray(X) 
        self.clf.fit(X, y) 
 
    # Predict the output class for the input datapoint 
    def _fit(self, X): 
        X = np.asarray(X) 
        return self.clf.predict(X) 
 
    # Encode the labels (convert words to numbers) 
    def _encodeLabels(self, labels_words): 
        self.le.fit(labels_words) 
        return np.array(self.le.transform(labels_words), 
         dtype=np.float32) 
 
    # Classify the input datapoint 
    def classify(self, X): 
        labels_nums = self._fit(X) 
        labels_words = self.le.inverse_transform([int(x) for x in labels_nums]) 
        return labels_words 
