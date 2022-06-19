from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def activation(net):
    return 1/(1+np.exp(-net))

def train(X,t,nepochs=200,n=0.5,test_size=0.3,val_size=0.3,seed=0):
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size,random_state=seed)
    X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=val_size,random_state=seed)

    train_accuracy = []
    val_accuracy = []
    nfeatures = X.shape[1]
    np.random.seed(seed)
    w = 2*np.random.uniform(size=(nfeatures,)) - 1
    
    for epoch in range(nepochs):
        y_train2 = X_train2.apply(lambda x: activation(np.dot(w,x)),axis=1)
        y_val = X_val.apply(lambda x: activation(np.dot(w,x)),axis=1)

        train_accuracy.append(sum(t_train2 == np.round(y_train2))/len(t_train2))
        val_accuracy.append(sum(t_val == np.round(y_val))/len(t_val))
                
        for j in range(len(w)):
            w[j] -= n*np.dot((y_train2 - t_train2)*y_train2*(1-y_train2),X_train2.iloc[:,j])
            
    results = pd.DataFrame({"epoch": np.arange(nepochs)+1, 'train_accuracy':train_accuracy,'val_accuracy':val_accuracy,
                            "n":n,'test_size':test_size,'val_size':val_size,'seed':seed
                           }).set_index(['n','test_size','val_size','seed'])
    return w,X_test,t_test,results

def evaluate_baseline(t_test,t_train2,t_val):
    frac_max_class = None
    accuracy_test = None
    accuracy_train2 = None
    accuracy_val = None
    ## BEGIN SOLUTION
    counts = t_train2.value_counts()
    frac_max_class = counts.iloc[0]/counts.sum()
    prediction = counts.index[0]
    accuracy_test = sum(t_test == prediction)/len(t_test)
    accuracy_train2 = sum(t_train2 == prediction)/len(t_train2)
    accuracy_val = sum(t_val == prediction)/len(t_val)
    ## END SOLUTION
    return frac_max_class,accuracy_test,accuracy_train2,accuracy_val

def predict(w,X,threshold=0.5):
    y = None
    ## BEGIN SOLUTION
    y = X.apply(lambda x: int(activation(np.dot(w,x)) > threshold),axis=1)
    ## END SOLUTION
    return y

def confusion_matrix(t,y,labels):
    cm = pd.DataFrame(columns=labels,index=labels)
    # actual is on the rows, pred on the columns
    ## BEGIN SOLUTION
    for actual in cm.index:
        for pred in cm.columns:
            cm.loc[actual,pred] = sum( (t==actual) & (y==pred) )
    ## END SOLUTION
    return cm

def evaluation(cm,positive_class=1):
    stats = {}
    ## BEGIN SOLUTION
    classes = list(cm.index)
    classes.remove(positive_class)
    tp = cm.loc[positive_class,positive_class]
    tn = fp = fn = 0
    for negative_class in classes:
        tn += cm.loc[negative_class,negative_class]
        fp += cm.loc[negative_class,positive_class]
        fn += cm.loc[positive_class,negative_class]
    stats['accuracy'] = (tp+tn)/(tp+tn+fp+fn)
    stats['sensitivity/recall'] = tp/(tp+fn)
    stats['specificity'] = tn/(tn+fp)
    stats['precision'] = tp/(tp+fp)
    stats['F1'] = 2*stats['precision']*stats['sensitivity/recall']/(stats['precision']+stats['sensitivity/recall'])
    ## END SOLUTION
    return stats

def importance(X,t,seeds):
    importances = pd.Series(np.zeros((X.shape[1],)),index=X.columns)
    ## BEGIN SOLUTION
    for seed in seeds:
        w,X_test,t_test,results = train(X,t,seed=seed)
        iw = np.sqrt(w**2)
        iw = pd.Series(iw/max(iw),index=X.columns)
        #print(iw)
        importances += iw
    importances = importances/len(seeds)
    ## END SOLUTION
    return importances
