import matplotlib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, KFold, cross_val_score, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import *
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFE
import xgboost as xgb
from scipy.stats import randint, expon, reciprocal, uniform
import time

from Preprocessing import preprocessing
from missingValues import missingValues

matplotlib.use('tkagg')

# Lecture des fichiers CSV
train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')


# TODO regarder importance des variables avec regression logistique et randomforest
# TODO SVM
# TODO tuner boosting
# TODO comparer accuracy sans tuner et avec
# TODO calculer moyenne et variance sur 5 itérations du modele

def dropRemainingMissingValues(train):
    print("Remaining missing values :  " + str(len(train[train.isna().sum(axis=1)>0])) + " / " + str(len(train)))
    train.drop(train[train.isna().sum(axis=1) > 0].index, inplace=True)
    print("After :  " + str(len(train)))


# Graphiques
def graph(data, x, y, type="strip"):
    if type == "strip":
        sns.stripplot(data=data, x=x, y=y, linewidth=0.1, s=1)
    if type == "count":
        data["t"] = data[x] + 2 * data[y]
        sns.countplot(data=data, x="t")
        data.drop("t", axis=1, inplace=True)

    plt.show()



#TODO Missing Values Preprocessing



# Random forest feature importante
def randomForest(train_process, y, test_process):
    print("########  RANDOMFOREST ########")
    """parameters = { 
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }
    rfc = RandomForestClassifier()
    search = GridSearchCV(estimator=rfc, param_grid=parameters, cv=5, scoring="accuracy", verbose=0).fit(train_process, y)
    model = search.best_estimator_
    print(search.best_params_)"""
    #### MEILLEUR HYPERPARAMETRE ####
    # criterion: entropy
    # max_depth: 8
    # max_features: sqrt
    # n_estimators: 200
    model = RandomForestClassifier(max_depth=8, criterion='entropy', max_features='sqrt', n_estimators=200)
    model.fit(train_process, y)
    """for i in range(model.n_features_in_):
        print(model.feature_names_in_[i] + "  :  " + str(model.feature_importances_[i]))"""
    pred = model.predict(test_process)
    sub = pd.read_csv("./Data/sample_submission.csv")
    sub['Transported'] = pred.astype(bool)
    sub.to_csv("./Data/submit.csv", index=False)


def xGBoost(train_process, y, test_process):
    print("########  XGBOOST ########")
    """boosted_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [4, 8, 12],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.5, 0.75, 1]
        }
    booster = xgb.XGBClassifier(objective= 'binary:logistic', eval_metric='logloss', random_state=0)
    search = GridSearchCV(estimator=booster, param_grid=boosted_grid, n_jobs=-1, cv=5, scoring='f1', verbose=0).fit(train_process, y)
    model = search.best_estimator_
    print(search.best_params_)"""
    #### MEILLEUR HYPERPARAMETRE #####
    # learning_rate =  0.05
    # max_depth = 4
    # n_estimators = 100
    # booster = gbtree
    # subsample = 1
    # min_child_weight = 1
    # gamma = 1
    model = xgb.XGBClassifier(objective= 'binary:logistic', eval_metric='logloss', learning_rate=0.05, max_depth=4, n_estimators=100, booster='gbtree', subsample=1, min_child_weight=1, gamma=1)
    model.fit(X_train, y_train)
    """for i in range(model.n_features_in_):
        print(model.feature_names_in_[i] + "  :  " + str(model.feature_importances_[i]))"""
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict(X_test))
    plt.plot(fpr,tpr,label="xgBoost="+str(auc))
    print(classification_report(y_test, model.predict(X_test)))
    pred = model.predict(test_process)
    sub = pd.read_csv("./Data/sample_submission.csv")
    sub['Transported'] = pred.astype(bool)
    sub.to_csv("./Data/submit.csv", index=False)
    
def SVM(train_process, y, test_process):
    print("########  SVM ########")
    ##### DETERMINATION MEILLEUR HYPERPARAMETRE #####
    """parameters = {
        'C':[0.001, 0.01, 0.1, 1, 10, 100],
        'kernel':['linear']
    }
    svc = SVC()
    grid = GridSearchCV(estimator = svc, param_grid = parameters, scoring = 'accuracy', cv = 5, verbose=0, n_jobs=-1)
    grid.fit(train_process, y)
    print('Parameters that give the best results :','\n\n', (grid.best_params_))
    svc = grid.best_estimator_"""
    #### MEILLEUR HYPERPARAMETRE #####
    # C = 1
    # kernel = "linear"
    svc = SVC(kernel="linear", C=1, probability=True)
    svc.fit(X_train,y_train)
    fpr, tpr, _ = roc_curve(y_test, svc.predict(X_test))
    auc = roc_auc_score(y_test, svc.predict(X_test))
    plt.plot(fpr,tpr,label="SVM="+str(auc))
    print(classification_report(y_test, svc.predict(X_test)))
    pred = svc.predict(test_process)
    sub = pd.read_csv("./Data/sample_submission.csv")
    sub['Transported'] = pred.astype(bool)
    sub.to_csv("./Data/submit.csv", index=False)


def Logistic(train_process, y, test_process):
    print("########  LOGISTIC ########")
    """parameters = {  'penalty': ['l1','l2'],
                    'C': np.logspace(-3,3,7),
                    'solver': ['saga', 'liblinear'],
                    "max_iter": [1000, 10000, 100000]
                }
    search = GridSearchCV(LogisticRegression(), parameters, scoring='accuracy', n_jobs=-1, cv=5, verbose=0).fit(train_process, y)
    model = search.best_estimator_
    print(search.best_params_)
    """
    #### MEILLEUR HYPERPARAMETRE #####
    # C = 100
    # max_iter = 1000
    # penalty = l2
    # solver = saga
    
    model = LogisticRegression(C=100, max_iter=1000, penalty='l2', solver='saga')
    model.fit(X_train, y_train)
    """for i in range(model.n_features_in_):
        print(model.feature_names_in_[i] + "  :  " + str(model.coef_[0][i]))"""
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict(X_test))
    plt.plot(fpr,tpr,label="regression="+str(auc))
    print(classification_report(y_test, model.predict(X_test)))
    pred = model.predict(test_process)
    sub = pd.read_csv("./Data/sample_submission.csv")
    sub['Transported'] = pred.astype(bool)
    sub.to_csv("./Data/submit.csv", index=False)


##### PREPROCESSING DES DONNEES ######
y = train["Transported"].copy().astype(int)
train_process = preprocessing(train.copy())
y = train_process["Transported"].copy().astype(int)
train_process.drop("Transported", axis=1, inplace=True)
test_process = preprocessing(test.copy(), True)

X_train, X_test, y_train, y_test = train_test_split(train_process, y, test_size=0.2, random_state=42)

Logistic(train_process, y, test_process)
SVM(train_process, y, test_process)
xGBoost(train_process, y, test_process)
randomForest(train_process, y, test_process)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

"""
df1 = pd.DataFrame()
df1["precision"] = [0.74655, 0.78887, 0.7912, 0.7898, 0.79003, 0.78933, 0.7898, 0.79284]
df1["m_p"] = ["regression", "SVM", 'random', 'random', 'random', 'random', 'random', 'xgboost']

sns.stripplot(data=df1, x="m_p", y="precision", hue="m_p", dodge=False)
plt.show()
"""