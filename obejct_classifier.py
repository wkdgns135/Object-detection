from create_features import *
from training import *
from classify_data import *

class object_classifier():
    def __init__(self):
        self.kmeans = None
        self.centroids = None
        self.svm = None

    def fit(self, x, y):
        
        self.kmeans, self.centroids = FeatureExtractor().get_centroids(x, y)

        feature_map = extract_feature_map(x, y, self.kmeans, self.centroids) 
    
        labels_words = [x['label'] for x in feature_map]
        dim_size = feature_map[0]['feature_vector'].shape[1]
    
        X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map] 
        
        self.svm = ClassifierTrainer(X, labels_words) 

    def predict(self, input):
        tag = ImageClassifier(self.svm, self.kmeans, self.centroids).getImageTag(input)
        return tag

    def score(self, X, Y):
        score = 0
        for x, y in zip(X, Y):
            tag = self.predict(x)
            if(y == tag):
                score+=1

        score /= len(Y)
        return score