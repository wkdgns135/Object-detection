import os 
import sys 
import argparse 
import json 
 
import cv2 
import numpy as np 
from sklearn.cluster import KMeans 

class SIFTExtractor():
    def __init__(self):
        self.extractor = cv2.xfeatures2d.SIFT_create(sigma = 1)

    def compute(self, image): 
        if image is None: 
            print("Not a valid image")
            raise TypeError 
 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        kps, des = self.extractor.detectAndCompute(gray_image, None)
        return kps, des

# Vector quantization 
class Quantizer(object): 
    def __init__(self, num_clusters=1024): 
        self.num_dims = 128 
        self.extractor = SIFTExtractor() 
        self.num_clusters = num_clusters 
        self.num_retries = 10
 
    def quantize(self, datapoints): 
        # Create KMeans object 
        kmeans = KMeans(self.num_clusters, 
                        n_init=max(self.num_retries, 1), 
                        max_iter=10, tol=1.0) 
 
        # Run KMeans on the datapoints 
        res = kmeans.fit(datapoints) 
 
        # Extract the centroids of those clusters 
        centroids = res.cluster_centers_
 
        return kmeans, centroids 
 
    def normalize(self, input_data): 
        sum_input = np.sum(input_data) 
        if sum_input > 0: 
            return input_data / sum_input 
        else: 
            return input_data 
 
    # Extract feature vector from the image 
    def get_feature_vector(self, img, kmeans, centroids): 
        kps, fvs = self.extractor.compute(img) 
        fvs = np.array(fvs, dtype=np.double)
        labels = kmeans.predict(fvs) 
        fv = np.zeros(self.num_clusters) 
 
        for i, item in enumerate(fvs): 
            fv[labels[i]] += 1 
 
        fv_image = np.reshape(fv, ((1, fv.shape[0]))) 
        return self.normalize(fv_image)


class FeatureExtractor(object): 
    def extract_image_features(self, img): 
        # SIFT feature extractor 
        kps, fvs = SIFTExtractor().compute(img) 
 
        return fvs 
 
    # Extract the centroids from the feature points 
    def get_centroids(self, data, target, num_samples_to_fit=10): 
        kps_all = [] 
        count = 0 
        cur_label = ''
        for x, y in zip(data, target):
            if count >= num_samples_to_fit:
                if cur_label != y:
                    count = 0
                else :
                    continue
            count += 1

            if count == num_samples_to_fit:
                None
                # print("Built centroids for", y)

            cur_label = y
            x = resize_to_size(x, 150)

            num_dims = 128
            fvs = self.extract_image_features(x)
            kps_all.extend(fvs)

        kmeans, centroids = Quantizer().quantize(kps_all) 
        return kmeans, centroids 
 
    def get_feature_vector(self, img, kmeans, centroids): 
        return Quantizer().get_feature_vector(img, kmeans, centroids) 
 
 
def extract_feature_map(data, kmeans, centroids): 
    features = []
    for x in data:
        temp_dict = {}
        x = resize_to_size(x, 150)
        temp_dict['feature_vector'] = FeatureExtractor().get_feature_vector(x, kmeans, centroids) 
        if temp_dict['feature_vector'] is not None: 
            features.append(temp_dict) 
    return features 
 
# Resize the shorter dimension to 'new_size' 
# while maintaining the aspect ratio 
def resize_to_size(input_image, new_size=150): 
    h, w = input_image.shape[0], input_image.shape[1] 
    ds_factor = new_size / float(h) 
 
    if w < h: 
        ds_factor = new_size / float(w) 
 
    new_size = (int(w * ds_factor), int(h * ds_factor)) 
    return cv2.resize(input_image, new_size) 
 