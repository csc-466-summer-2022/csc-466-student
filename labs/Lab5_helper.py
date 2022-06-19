import copy
import json

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

def get_learner(X,y,max_depth=10):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)

## BEGIN SOLUTION
from sklearn.utils import resample
def boostrap_sample(X,y):
    X, y = resample(X, y)
    return X,y
## END SOLUTION

def make_trees(X,y,ntrees=100,max_depth=10):
    trees = []
    for i in range(ntrees):
        ## BEGIN SOLUTION
        Xsample, ysample = boostrap_sample(X,y)
        trees.append(get_learner(Xsample,ysample,max_depth=max_depth))
        ## END SOLUTION
        # Your solution here
        pass
        
    return trees

def make_prediction(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)

def make_trees_boost(Xtrain, Xval, ytrain, yval, max_ntrees=100,max_depth=2):
    #Xtrain, Xval, ytrain, yval = train_test_split(X,y,test_size=val_frac,shuffle=True)
    trees = []
    yval_pred = None
    ytrain_pred = None
    train_RMSEs = [] # the root mean square errors for the validation dataset
    val_RMSEs = [] # the root mean square errors for the validation dataset
    ytrain_orig = copy.deepcopy(ytrain)
    for i in range(max_ntrees):
        ## BEGIN SOLUTION
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(Xtrain,ytrain)
        ytrain = ytrain-model.predict(Xtrain)
        if yval_pred is None:
            yval_pred = model.predict(Xval)
            mse = np.sqrt(np.sum((yval_pred-yval)**2)/len(yval))
            val_RMSEs.append(mse)
            
            ytrain_pred = model.predict(Xtrain)
            mse = np.sqrt(np.sum((ytrain_pred-ytrain_orig)**2)/len(ytrain_orig))
            train_RMSEs.append(mse)
        else:
            yval_pred = yval_pred + model.predict(Xval)
            mse = np.sqrt(np.sum((yval_pred-yval)**2)/len(yval))
            val_RMSEs.append(mse)
            
            ytrain_pred = ytrain_pred + model.predict(Xtrain)
            mse = np.sqrt(np.sum((ytrain_pred-ytrain_orig)**2)/len(ytrain_orig))
            train_RMSEs.append(mse)
        trees.append(model)
        ## END SOLUTION
        # Your solution here
        pass
        
    return trees,train_RMSEs,val_RMSEs

def cut_trees(trees,val_RMSEs):
    ## BEGIN SOLUTION
    dt = pd.Series(val_RMSEs)
    ix = dt.idxmin()
    trees = trees[:(ix+1)]
    ## END SOLUTION
    # Your solution here that finds the minimum validation score and uses only the trees up to that
    return trees

def make_prediction_boost(trees,X):
    tree_predictions = []
    for tree in trees:
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).sum().values.flat)

