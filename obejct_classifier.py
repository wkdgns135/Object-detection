from create_features import *

class object_classifier():
    def __init__(self, estimator, kmeans, centorids):
        self.kmeans = kmeans
        self.centroids = centorids
        self.estimator = estimator
        self.pred = None
        # hyperparameters

    def fit(self, data, target):
        self.estimator = self.estimator.fit(data, target)

    def predict(self, input):
        input = np.asarray(input)
        return self.estimator.predict(input)

    def score(self, data, target):
        score = 0
        pred = self.predict(data)
        for y, y_pred in zip(target , pred):
            if(y == y_pred):
                score+=1
        score /= len(target)
        return score
    
    def get_params(self,deep=False):
        return {'C':self.C}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self