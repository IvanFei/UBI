# -*- coding: utf-8 -*-

import numpy as np
import sklearn
import scipy as sp

# model 
from sklearn import linear_model, svm, ensemble, neighbors
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from xgboost import XGBRegressor

def model(X, Y):
    
#    MLA = [
#    # ensemble Model
#    ensemble.AdaBoostRegressor(),
#    ensemble.GradientBoostingRegressor(),
#    ensemble.ExtraTreesRegressor(), 
#    
#    #GLM
#    linear_model.SGDRegressor(),
#    
#    #SVM
#    svm.NuSVR(),
#    svm.SVR(),
#    
#    #xgboost
#    XGBRegressor()
#    ]

    # data splite
    split = model_selection.ShuffleSplit(n_splits=10, random_state=0)
    
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    #grid_criterion = ['gini', 'entropy']
    grid_seed = [0]
    n_jobs = 2

    vote_est = [
        #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
        ('ada', ensemble.AdaBoostRegressor()),
        ('bc', ensemble.BaggingRegressor(n_jobs=n_jobs)),
        ('gbc', ensemble.GradientBoostingRegressor()),
        ('rfc', ensemble.RandomForestRegressor(n_jobs=n_jobs)),
        ('etc',ensemble.ExtraTreesRegressor(n_jobs=n_jobs)),
        
        #SVM: http://scikit-learn.org/stable/modules/svm.html
        ('svc', svm.SVR()),
        
        #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        ('xgb', XGBRegressor())
    
    ]
    
    grid_param = [
                    [{
                    #AdaBoostRegressor - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
                    'n_estimators': grid_n_estimator, #default=50
                    'learning_rate': grid_learn, #default=1
                    #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
                    'random_state': grid_seed
                    }],
           
        
                    [{
                    #BaggingRegressor - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
                    'n_estimators': grid_n_estimator, #default=10
                    'max_samples': grid_ratio, #default=1.0
                    'random_state': grid_seed
                     }],
    
    
                    [{
                    #GradientBoostingRegressor - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
                    #'loss': ['deviance', 'exponential'], #default=’deviance’
                    'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
                    'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
                    #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
                    'max_depth': grid_max_depth, #default=3   
                    'random_state': grid_seed
                     }],
    
        
                    [{
                    #RandomForestRegressor - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
                    'n_estimators': grid_n_estimator, #default=10
                    #'criterion': grid_criterion, #default=”gini”
                    'max_depth': grid_max_depth, #default=None
                    'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
                    'random_state': grid_seed
                     }],
        
                       [{
                    #ExtraTreesRegressor - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
                    'n_estimators': grid_n_estimator, #default=10
                    #'criterion': grid_criterion, #default=”gini”
                    'max_depth': grid_max_depth, #default=None
                    'random_state': grid_seed
                     }],
        
               
        
                    [{
                    #SVR - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
                    #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
                    #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [1,2,3,4,5], #default=1.0
                    'gamma': grid_ratio, #edfault: auto
                     }],
    
        
                    [{
                    #XGBRegressor - http://xgboost.readthedocs.io/en/latest/parameter.html
                    'learning_rate': grid_learn, #default: .3
                    'max_depth': [1,2,4,6,8,10], #default 2
                    'n_estimators': grid_n_estimator, 
                    'seed': grid_seed  
                     }]
    ]

    for rlf, param in zip(vote_est, grid_param):
        
        best_search = model_selection.GridSearchCV(estimator=rlf[1], param_grid=param, cv=split, scoring='neg_mean_squared_error', n_jobs=n_jobs)
        best_search.fit(X, Y)
        
        best_param = best_search.best_params_
        bestIndex = best_search.best_index_
        trainBestScore = best_search.cv_results_['mean_train_score'][bestIndex]
        testBestScore = best_search.best_score_
        print('The best parameter for {} is {}.'.format(rlf[1].__class__.__name__, best_param))
        print('The train best score {:.3f}'.format(trainBestScore))
        print('The test best score {:.3f}'.format(testBestScore))
        
        rlf[1].set_params(**best_param)
        
    return vote_est
        
        