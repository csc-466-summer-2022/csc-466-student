import copy
import json

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

def entropy(y):
    e = None
    # YOUR SOLUTION HERE
    ## BEGIN SOLUTION
    counts = y.value_counts()
    counts = counts.fillna(0)
    probs = counts/sum(counts)
    probs_log = copy.deepcopy(probs)
    probs_log[probs==0] = 1
    logs = np.log2(probs_log)
    e = sum(-probs*(logs))
    ## END SOLUTION
    return e

def gain(y,x):
    g = 0
    # YOUR SOLUTION HERE
    ## BEGIN SOLUTION
    for f in x.unique():
        mask = x == f
        ysplit = y.loc[mask]
        esplit = entropy(ysplit)
        g += len(ysplit)/len(y)*esplit
    ## END SOLUTION
    return entropy(y) - g

def gain_ratio(y,x):
    # YOUR SOLUTION HERE
    ## BEGIN SOLUTION
    g = gain(y,x)
    ## END SOLUTION
    return g/entropy(y)

def select_split(X,y):
    col = None
    gr = None
    ## BEGIN SOLUTION
    gain_ratios = {}
    for column in X.columns:
        gain_ratios[column] = gain_ratio(y,X[column])
    gain_ratios = pd.Series(gain_ratios)
    col,gr = gain_ratios.idxmax(),gain_ratios.max()
    ## END SOLUTION
    # YOUR SOLUTION HERE
    return col,gr

def make_tree(X,y):
    tree = {}
    # Your solution here
    ## BEGIN SOLUTION
    if len(y.unique()) == 1:
        return y.unique()[0]
    if len(X.columns) == 0: # nothing to check and we've used all variables
        return y.value_counts().idxmax()
    split_var,gain = select_split(X,y)
    # nothing to gain
    if gain <= 0.000001:
        return y.value_counts().idxmax()
    tree[split_var] = {}
    for xv in X[split_var].unique():
        mask = X[split_var] == xv
        Xsubset = X.loc[mask]
        ysubset = y.loc[mask]
        tree[split_var][xv] = make_tree(Xsubset.drop(split_var,axis=1),ysubset)
    ## END SOLUTION
    return tree

# if you want to print like me :)
def print_tree(tree):
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            return int(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))

def generate_rules(tree):
    rules = []
    # Your solution here
    ## BEGIN SOLUTION
    if type(tree) != dict:
        return [[tree]]
    for key in tree.keys():
        for xv in tree[key]:
            new_rules = generate_rules(tree[key][xv])
            for rule in new_rules:
                rules.append([(key,xv)]+rule)
    ## END SOLUTION
    return rules

def split_col(x, y):
    x2 = list(x.copy())
    save_x = x.copy()
    x2.sort()
    splits = []
    for i in range(0, len(x2)-1):
        splits.append((x2[i] + x2[i+1]) / 2)
    for split in splits:
        x = x.apply(lambda x: True if x < split else False)
        g = gain_ratio(y, x)
        if g > gr:
            gr, col = g, c
            
        x = save_x.copy()
    return g, c

def select_split2(X,y):
    col = None
    gr = float("-inf")
    best_split = None
    for c in X.columns:
        if X[c].dtype in ['int64','float64']:
            
            save_x = X[c].copy()
            x2 = list(X[c].copy())
            x2.sort()
            
            splits = []
            for i in range(0, len(x2)-1):
                splits.append((x2[i] + x2[i+1]) / 2)
                
            for split in splits:
                save_x = save_x.apply(lambda x: 'True' if x < split else 'False')
                g = gain_ratio(y, save_x)
                if g > gr:
                    gr, col, best_split = g, c, split
                save_x = X[c].copy()
        else:
            g = gain_ratio(y, X[c])
            if g > gr:
                gr, col, best_split = g, c, None
    newname = col
    if best_split:
        newname = '%s<%.2f'%(col, best_split)
        X.rename(columns={col:newname}, inplace=True)
        X[newname] = X[newname].apply(lambda x: "True" if x < best_split else "False")
    return newname ,gr

def make_tree2(X,y,min_split_count=5):
    tree = {}
    if len(X.columns) == 0:
        return y.value_counts().idxmax()
    if len(y.unique()) == 1:
        return y.unique()[0]
    if len(X) < min_split_count:
        return y.value_counts().idxmax()
    col, gr = select_split2(X, y)
    if gr < 0.001:
        return y.value_counts().idxmax()
    tree[col] = {}
    for ux in X[col].unique():
        tree[col][ux] = {}
        y2 = y[X[col] == ux]
        X2 = X.copy()
        X2 = X2[X2[col] == ux]
        X2 = X2.drop(col, axis=1)
        tree[col][ux] = make_tree2(X2, y2, min_split_count)
    return tree

def make_prediction(rules,x,default):
    # Your solution here
    ## BEGIN SOLUTION
    for rule in rules:
        matches = True
        for cond in rule[:-1]:
            split_value,xv = cond
            if "<" not in split_value:
                if x.loc[split_value] != xv:
                    matches = False
                    break
            else:
                col,cutpoint = split_value.split("<")
                cutpoint = float(cutpoint)
                if xv == 'True':
                    if x.loc[col] >= cutpoint:
                        matches = False
                        break
                else:
                    if x.loc[col] < cutpoint:
                        matches = False
                        break
        if matches:
            return rule[-1]
    ## END SOLUTION
    return(default)
