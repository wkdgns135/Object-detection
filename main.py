import pandas as pd
from obejct_classifier import *

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


def train_test_split(categories, n = 30):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for category in categories:
        fileName = os.listdir("101_ObjectCategories/" + category)
        count = 0
        for name in fileName:
            img = cv2.imread("101_ObjectCategories/"+category+"/" +name)
            if(count < n):
                train_x.append(img)
                train_y.append(category)
                count+=1
            else:
                test_x.append(img)
                test_y.append(category)

    return train_x, test_x, train_y, test_y

if __name__ == "__main__":
    categories = ["ant", "chair", "bass","crab","emu"]

    train_x, test_x, train_y, test_y = train_test_split(categories, n=30)
    
    
    model = DecisionTreeClassifier()
    oc = object_classifier(model)
    oc.fit(train_x, train_y)
    print(oc.score(test_x, test_y))

    # for c in range(1,100):
    #     oc.fit(train_x, train_y)
    #     # print(oc.predict(test_x[66]))
    #     print(c,oc.score(test_x, test_y))
        