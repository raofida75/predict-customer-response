# import required libraries
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
#model evaluation
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error



def train(model, X_train, y_train, X_test, y_test, classifier=True):
    
    """ 
    Train the given model on the training data, and evaluate the model 
    on the test data.
    INPUT:
        model: model to be trained
        X_train: features used to train the model
        y_train: target variables for training
        X_test: features used to evaluate the model
        y_test: target variables for testing
    """
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = np.round(end - start,2)
    print(f'{model.__class__.__name__} took {train_time} seconds to train on the data.')
    ypred = model.predict(X_test)
    
    if classifier == True:
        acc, auc, f1 = evaluate_classifier(model, X_test, y_test, ypred)
        return acc, auc, f1
    else:
        evaluate_regressor(y_test, ypred)    



def evaluate_classifier(model, X_test, y_test, ypred):
    
    """ 
    Evaluate the performance of the model on the test dataset
    INPUT:
        X_test: features used to evaluate the model
        y_test: target variables for testing
        ypred: predictions on the test data
    OUTPUT:
        f1: F1 score of the model, useful if there is a class imbalance.
        auc: Roc Auc score of the model, useful if there is a class imbalance.
        acc: Accuracy of the model, will only be considered if the classes
        are balanced.
    """

    acc = accuracy_score(y_test, ypred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_test, ypred)
    print()
    print('Accuracy: {:.4f}'.format(acc))
    print('AUC score: {:.4f}'.format(auc))
    print('F1 score: {:.4f}'.format(f1))
    print()
    print(classification_report(y_test, ypred))
    return acc, auc, f1



def evaluate_regressor(y_test, ypred):    
    """ 
    Evaluate the performance of the regressor on the test dataset
    INPUT:
        X_test: features used to evaluate the model
        y_test: target variables for testing
        ypred: predictions on the test data
    OUTPUT:
        r2: F1 score of the model, useful if there is a class imbalance.
        auc: Roc Auc score of the model, useful if there is a class imbalance.
        acc: Accuracy of the model, will only be considered if the classes
        are balanced.
    """
    r2 = r2_score(y_test, ypred)
    mse = mean_squared_error(y_test, ypred)
    rmse = np.sqrt(mse)
    print()
    print('r2 score: {:.4f}'.format(r2))
    print('Mean Squared Error: {:.4f}'.format(mse))
    print('Root Mean Squared Error: {:.4f}'.format(rmse))
    print()



param_grid = {
    'rfc': {
        'randomforestclassifier__n_estimators':[i for i in range(100,501,100)],
        'randomforestclassifier__max_depth': [None,2,5,8],
        'randomforestclassifier__min_samples_leaf': [1,3,6,10]
    },
    'gbc': {
        'gradientboostingclassifier__learning_rate':[0.1, 1, 10],
        'gradientboostingclassifier__min_samples_leaf': [1,3,6,10],
        'gradientboostingclassifier__n_estimators':[i for i in range(100,301,100)]
    }
}

def grid_pipeline(X_train, y_train, X_test, y_test, scoring, params=param_grid):
    
    pipelines = {
        'rfc': make_pipeline(RandomForestClassifier(random_state=0)),
        'gbc': make_pipeline(GradientBoostingClassifier(random_state=0))
    }
    fitted_model = {}
    for key, model in pipelines.items():
        grid_model = GridSearchCV(model, param_grid=params[key], cv=5, n_jobs=-1, scoring=scoring)
        try:
            print('Started training for {} model'.format(key))
            grid_model.fit(X_train, y_train)
            fitted_model[key] = grid_model
            print('Successfully fitted {} model'.format(key))
        except NotFittedError as e:
            print(repr(e))
    scores = evaluate_grid_pipeline(fitted_model, scoring, X_test, y_test)
    return fitted_model, scores

def evaluate_grid_pipeline(fitted_model, scoring, X_test, y_test): 
    scores = []           
    for key, f_model in fitted_model.items():
        ypred = f_model.predict(X_test)
        if scoring == 'accuracy':
            score = accuracy_score(y_test, ypred)
        elif scoring == 'roc_auc':
            score = roc_auc_score(y_test, f_model.predict_proba(X_test)[:,1])
        else:
            score = f1_score(y_test, ypred)
        scores.append(score)
    return scores

    