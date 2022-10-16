import os
from matplotlib import test
import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.compose import *
from sklearn.pipeline import *
from sklearn.metrics import *
from sklearn.impute import *
from sklearn.multioutput import *
from catboost import CatBoostClassifier
import src.config as cfg


def build_real_pipe():
    return Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())])

def build_cat_pipe():
    return Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])

def build_preprocess_pipe():
    return ColumnTransformer(transformers=[
    ('real_cols', build_real_pipe(), cfg.REAL_COLS),
    ('cat_cols', build_cat_pipe(), cfg.CAT_COLS),
    ('ohe_cols', 'passthrough', cfg.OHE_COLS)])

def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
            iterations=200,
            depth=8,
            silent=True,
            l2_leaf_reg=2.0,
            learning_rate=0.0001,
            early_stopping_rounds=100,
            loss_function="MultiLogloss")

def build_model_train_pipe(model: CatBoostClassifier, preprocess_pipe):
    return Pipeline([
    ('preprocess', preprocess_pipe),
    ('model', model)])


def train_model(train_data: pd.DataFrame, target_data: pd.DataFrame):
    model = build_model()
    preprocess_pipe = build_preprocess_pipe()
    train_pipe = build_model_train_pipe(model, preprocess_pipe)

    result = cross_validate(
    estimator=train_pipe,
    X=train_data,
    y=target_data,
    scoring='roc_auc',
    cv=5,
    n_jobs=8,
    return_estimator=True,
    return_train_score = True)

    return (get_scores(result), get_model(result));


def predict(test_data: pd.DataFrame, model: CatBoostClassifier):
    preprocess_pipe = build_preprocess_pipe()
    data = preprocess_pipe.fit_transform(test_data)
    return model.predict(data)


def get_model(result):
    est = result['estimator']
    pipe = est.pop()
    return pipe[1]

def get_scores(result):
    return {'test_score': list(result['test_score']), 'train_score': list(result['train_score'])}