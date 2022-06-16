# -*- coding: utf-8 -*-
import pandas as pd
from create_features_2 import *
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


def data_import():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    categories =[]

    index = 0
    train_dir_name = 'Data/Train/'
    for dirname in os.listdir(train_dir_name):
        categories.append(dirname)
        img_dir_name = train_dir_name + dirname
        for fname in os.listdir(img_dir_name):
            img = cv2.imread(img_dir_name+'/'+fname)
            train_x.append(img)
            train_y.append(index)
        index+=1

    index = 0
    test_dir_name = 'Data/Test/'
    for dirname in os.listdir(test_dir_name):
        img_dir_name = test_dir_name + dirname
        for fname in os.listdir(img_dir_name):
            img = cv2.imread(img_dir_name+'/'+fname)
            test_x.append(img)
            test_y.append(index)
        index+=1

    return train_x, test_x, train_y, test_y, categories
            
def data_processing(data, target = None, kmeans = None, centroids = None):
    if(target != None):
        kmeans, centroids = FeatureExtractor().get_centroids(data, target)

    feature_map = extract_feature_map(data, kmeans, centroids)
    dim_size = feature_map[0]['feature_vector'].shape[1]

    x = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map] 

    if(target != None):
        return x, kmeans, centroids
    return x

def hyperparameter_tuning(data, target):
    model_to_set = OneVsRestClassifier(svm.SVC())

    kernel = ['linear','rbf','poly']
    C=[1, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
    # gamma = [round(0.1*i,1) for i in range(1, 11)]
    gamma = [1, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
    params = {'estimator__kernel':kernel, 'estimator__C':C, 'estimator__gamma':gamma}
    gs = GridSearchCV(estimator=model_to_set, param_grid=params, n_jobs=-1, cv=3)
    gs.fit(data, target)
    
    # print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)

    return gs.best_estimator_

def score(pred, target, categories):
    pred = [pred[i: i + 20] for i in range(0, len(pred), 20)]
    target = [target[i: i + 20] for i in range(0, len(target), 20)]

    for pred_Y, Y, category in zip(pred, target, categories):
        score = 0
        for pred_y, y in zip(pred_Y, Y):
            if(pred_y == y):
                score +=1
        print(category+" class는 20개중 ",score,"개 맞추었으며 score는 ",(score / 20),"입니다.")

if __name__ == "__main__":
    train_x, test_x, train_y, test_y, categories = data_import()
    train_x , kmeans, centroids = data_processing(train_x, train_y)
    test_x = data_processing(test_x, kmeans=kmeans, centroids=centroids)

    best_model = hyperparameter_tuning(train_x, train_y)
    # best_params ={'C': 1, 'gamma': 100, 'kernel': 'rbf'}
    # best_model = OneVsRestClassifier(svm.SVC(**best_params))

    best_model.fit(train_x, train_y)
    
    pred = best_model.predict(test_x)
    score(pred, test_y, categories)
    print("전체 스코어:", best_model.score(test_x, test_y))