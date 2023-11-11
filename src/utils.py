import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            mdl = pickle.load(f)
            return mdl
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    reports = {}
    try:
        for model in models:
            cv = GridSearchCV(models[model],cv = 3,param_grid=param[model])
            cv.fit(X_train,y_train)
            y_pred = cv.predict(X_test)
            score = r2_score(y_test,y_pred)
            reports[model] = score
        return reports
    except Exception as e:
        raise CustomException(e,sys)
    