from create_features import *
from training import *
from classify_data import *

class object_classifier():
    def __init__(self, categories):
        self.cls = []
        self.kmeans = None
        self.centroids = None
        self.svm = None
        for cls in categories:
            self.cls.append([cls, '101_ObjectCategories/'+cls])

    def fit(self, k = 32):
        input_map = []
        for cls in self.cls:
            assert len(cls) >= 2, "Format for classes is `<label> file`" 
            label = cls[0] 
            input_map += load_input_map(label, cls[1])
        
        self.kmeans, self.centroids = FeatureExtractor().get_centroids(input_map, k)

        feature_map = extract_feature_map(input_map, self.kmeans, self.centroids) 
    
        labels_words = [x['label'] for x in feature_map] 
        dim_size = feature_map[0]['feature_vector'].shape[1] 
    
        X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map] 
        
        self.svm = ClassifierTrainer(X, labels_words) 

    def predict(self, input):
        tag = ImageClassifier(self.svm, self.kmeans, self.centroids).getImageTag(input)
        return tag