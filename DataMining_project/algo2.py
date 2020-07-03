import matplotlib
import numpy as np
import pandas as pd
import os, sys
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
import time

def load_dataset():
    # train_path = "data/train_data_v1.csv"
    # data = pd.read_csv(train_path,sep=" ")
    # data['used_time'] = (pd.to_datetime(data['creatDates'], format='%Y-%m-%d', errors='coerce') - 
    #                         pd.to_datetime(data['regDates'], format='%Y-%m-%d', errors='coerce')).dt.days
    # train = data[["bodyType","brand","fuelType","gearbox","kilometer",
    #             'model', 'notRepairedDamage', 'power', 'regDate',
    #             'v_0', 'v_1', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14',
    #             'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 
    #             'name_count','used_time']]
    # label = data["price"]
    # x_train, x_test, y_train, y_test = train_test_split(train, label, test_size=0.3)

    paths = ["dataset/x_train.csv", "dataset/x_test.csv","dataset/y_train.csv", "dataset/y_test.csv"]
    x_train, x_test, y_train, y_test = [joblib.load(x) for x in paths]
    #std = StandardScaler()
    #preprocessing.scale(x_train)
    #preprocessing.scale(x_test)

    return x_train, x_test, y_train, y_test

def train_decision_tree_regressor(x_train, y_train):
    dr = tree.DecisionTreeRegressor()
    dr.fit(x_train, y_train)
    return dr

def train_random_forest_regressor(x_train, y_train):
    #rfr = RandomForestRegressor(n_estimators=80, max_depth=25 )
    rfr = RandomForestRegressor(n_estimators=80,)
    rfr.fit(x_train, y_train)
    return rfr
def train_LightGBM(x_train, y_train):
    clf = LGBMRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        boosting_type= 'gbdt',
        objective = 'regression_l1',
        max_depth = -1,
        num_leaves=31,
        min_child_samples = 20,
        feature_fraction = 0.8,
        bagging_freq = 1,
        bagging_fraction = 0.8,
        lambda_l2 = 2,
        random_state=2020,
    )
    clf.fit(x_train, y_train)
    return clf

def train_XGBRegressor(x_train, y_train):
    gbm= XGBRegressor()
    gbm.fit(x_train, y_train)
    return gbm
def Weighted_method(test_pre1,test_pre2,w=[1/2,1/2]):
    Weighted_result = w[0]*pd.Series(test_pre1)+w[1]*pd.Series(test_pre2)
    return Weighted_result

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_dataset()
    if len(sys.argv)<3: count = len(x_train)
    else:
        count = int(sys.argv[2])
    x_train = x_train[:count]
    y_train = y_train[:count]
    t0 = time.time()
    print("fitting")
    if sys.argv[1] == "lgbm":
        drm = train_LightGBM(x_train, y_train)
        pkl_name = "train_LightGBM"

    if sys.argv[1] == "xgbt":
        drm = train_XGBRegressor(x_train, y_train)
        pkl_name = "XGBRegressor"
    if sys.argv[1] == "tree":
        drm = train_decision_tree_regressor(x_train, y_train)
        pkl_name = "train_decision_tree_regressor"
    if sys.argv[1] == "forest":
        drm = train_random_forest_regressor(x_train, y_train)
        pkl_name = "train_forest_regressor"

    t1 = time.time()
    print("excute_time: %f"%round(t1 - t0, ndigits=4))
    print("dumping model")
    joblib.dump(drm, "models/"+pkl_name+".pkl") 

    predict_test = drm.predict(x_test)
    print(mean_absolute_error(predict_test, y_test))
    print(mean_absolute_error(np.expm1(predict_test), np.expm1(y_test)))