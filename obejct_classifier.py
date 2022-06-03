from create_features import *
from training import *
from classify_data import *

class object_classifier():
    def __init__(self, C=1):
        self.kmeans = None
        self.centroids = None
        self.svm = None

        # hyperparameters
        self.C = C

    def fit(self, data, target):
        
        self.kmeans, self.centroids = FeatureExtractor().get_centroids(data, target)

        feature_map = extract_feature_map(data, target, self.kmeans, self.centroids) 
    
        labels_words = [x['label'] for x in feature_map]
        dim_size = feature_map[0]['feature_vector'].shape[1]
    
        X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map] 
        
        self.svm = ClassifierTrainer(X, labels_words, self.C) 

    def predict(self, input):
        tag = ImageClassifier(self.svm, self.kmeans, self.centroids).getImageTag(input)
        return tag

    def score(self, data, target):
        score = 0
        for x, y in zip(data, target):
            tag = self.predict(x)
            if(y == tag):
                score+=1

        score /= len(target)
        return score
    
    def get_params(self,deep=False):
        return {'C':self.C}

    def set_params(self, C):
        self.C = C