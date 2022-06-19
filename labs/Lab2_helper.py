import numpy as np

def compute_priors(y):
    priors = {}
    ## BEGIN SOLUTION
    keys = np.unique(y)
    for key in keys:
        priors[f"{y.name}={key}"] = sum(y==key)/len(y)
    ## END SOLUTION
    return priors

def specific_class_conditional(x,xv,y,yv):
    prob = None
    ## BEGIN SOLUTION
    num = sum((y==yv) & (x==xv))
    den = sum((y==yv))
    prob = num/den
    ## END SOLUTION
    return prob

def class_conditional(X,y):
    probs = {}
    ## BEGIN SOLUTION
    yname = y.name
    y_values = np.unique(y)
    for c in X.columns:
        c_values = np.unique(X[c])
        for xv in c_values:
            for yv in y_values:
                probs["%s=%s|%s=%s"%(c,xv,yname,yv)] = specific_class_conditional(X[c],xv,y,yv)
    ## END SOLUTION
    return probs

def posteriors(probs,priors,x):
    post_probs = {}
    ## BEGIN SOLUTION
    norm = 0
    for ypart in priors.keys():
        xparts = []
        prob = 1
        for c in x.index:
            xpart = "%s=%s"%(c,x.loc[c])
            xparts.append(xpart)
            key = "%s|%s"%(xpart,ypart)
            if key not in probs:
                prob = prob * 0
            else:
                prob = prob * probs[key]
        post_probs["%s|"%ypart+",".join(xparts)] = priors[ypart]*prob
        norm += priors[ypart]*prob
    for key in post_probs:
        if norm == 0:
            post_probs[key] = 1/len(list(priors.keys()))
        else:
            post_probs[key] = post_probs[key]/norm
    ## END SOLUTION
    return post_probs

def train_test_split(X,y,test_frac=0.5):
    inxs = list(range(len(y)))
    np.random.shuffle(inxs)
    X = X.iloc[inxs,:]
    y = y.iloc[inxs]
    Xtrain,ytrain,Xtest,ytest = None,None,None,None
    ## BEGIN SOLUTION
    clean_inxs = list(range(len(y)))
    train_inxs = clean_inxs[:round(len(y)*(1-test_frac))]
    test_inxs = clean_inxs[round(len(y)*(1-test_frac)):]
    Xtrain = X.iloc[train_inxs,:]
    ytrain = y.iloc[train_inxs]
    Xtest = X.iloc[test_inxs,:]
    ytest = y.iloc[test_inxs]
    ## END SOLUTION
    return Xtrain,ytrain,Xtest,ytest

def exercise_6(Xtrain,ytrain,Xtest,ytest):
    accuracy = None
    ## BEGIN SOLUTION
    probs = class_conditional(Xtrain,ytrain)
    priors = compute_priors(ytrain)
    test_posteriors = Xtest.apply(lambda x: posteriors(probs,priors,x),axis=1)
    predictions = np.zeros((len(ytest),))
    for i,row in enumerate(test_posteriors):
        max_key = None
        max_prob = -1
        for key in row.keys():
            if row[key] > max_prob:
                max_prob = row[key]
                max_key = key
        predictions[i] = max_key.split("|")[0].split("=")[1]
    accuracy = sum(predictions==ytest)/len(ytest)
    ## END SOLUTION
    return accuracy

def exercise_7(Xtrain,ytrain,Xtest,ytest, npermutations = 10):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
            ## BEGIN SOLUTION
            new_accuracy = exercise_6(Xtrain,ytrain,Xtest2,ytest)
            print(new_accuracy)
            importances[col] += orig_accuracy - new_accuracy
            ## END SOLUTION
        importances[col] = importances[col]/npermutations
    return importances

def exercise_8(Xtrain,ytrain,Xtest,ytest, npermutations = 20):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtrain2 = Xtrain.copy()
            Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values
            ## BEGIN SOLUTION
            new_accuracy = exercise_6(Xtrain2,ytrain,Xtest,ytest)
            importances[col] += orig_accuracy - new_accuracy
            ## END SOLUTION
        importances[col] = importances[col]/npermutations
    return importances
