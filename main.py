import pandas as pd
from create_features import *
# from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier,plot_importance
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV


def data_import(categories):
    data = []
    target = []
    index = 0
    for category in categories:
        fileName = os.listdir("101_ObjectCategories/" + category)
        for name in fileName:
            img = cv2.imread("101_ObjectCategories/"+category+"/" +name)
            data.append(img)
            target.append(index)
        index+=1
    return data, target
            
def data_processing(data):
    kmeans, centroids = FeatureExtractor().get_centroids(data, target)

    feature_map = extract_feature_map(data, kmeans, centroids) 
    dim_size = feature_map[0]['feature_vector'].shape[1]

    x = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map] 

    return x, kmeans, centroids

def train_test_split(data, target, n = 30):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    count =  0
    pre_y = ""
    for x, y in zip(data, target):
        if(pre_y != y):
            count=0
        if(count < 30):
            train_x.append(x)
            train_y.append(y)
        else:
            test_x.append(x)
            test_y.append(y)
        count+=1
        pre_y = y

    return train_x, test_x, train_y, test_y

def hyperparameter_tuning(data, target):
    model_to_set = OneVsRestClassifier(svm.SVC())

    kernel = ['linear','rbf','poly']
    C=[1, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
    gamma = [round(0.1*i,1) for i in range(1, 11)]
    params = {'estimator__kernel':kernel, 'estimator__C':C, 'estimator__gamma':gamma}
    gs = GridSearchCV(estimator=model_to_set, param_grid=params, n_jobs=-1, cv=3)
    gs.fit(data, target)
    
    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)

    return gs.best_params_


if __name__ == "__main__":
    categories = ["accordion", "chair", "bass","crab","emu"]
    data, target = data_import(categories=categories)

    data , kmeans, centroids = data_processing(data)
    train_x, test_x, train_y, test_y = train_test_split(data, target)

    best_params = hyperparameter_tuning(data, target)
    # best_params = {'C': 1000, 'gamma': 0.1, 'kernel': 'rbf'}

    # best_model = OneVsRestClassifier(svm.SVC(**best_params))
    # best_model.fit(train_x, train_y)
    # print(best_model.score(test_x, test_y))