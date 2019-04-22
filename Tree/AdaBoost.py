from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd

def load_data():
    dataset = pd.read_csv("../train.csv", sep = ',', header = None)

    X = dataset.values[1:, 2:]
    Y = dataset.values[1:, 1]

    size = X.shape
    X = np.asarray(X).astype(float)
    Y = np.reshape(Y, [size[0]]).astype(int)
    
    X_mean = np.mean(X, axis = 0)
    X_max = np.max(X, axis = 0)
    X_min = np.min(X, axis = 0)

    X_ = (X - X_mean)/((X_max == X_min).astype(float) + X_max - X_min)
    X_ = np.asarray(X_).astype(float)

    X_tr, X_te, Y_tr, Y_te = train_test_split(X_, Y, test_size = 0.3, random_state = 100)
    return X_tr, X_te, Y_tr, Y_te


def train(X_train, X_test, Y_train, Y_test, tree_type = 'gini'):
    if tree_type == 'gini':
        tree_clf = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 3, min_samples_leaf = 5)
    elif tree_type == 'entropy':
        tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 100, max_depth = 3, min_samples_leaf = 5)
    else:
        tree_clf = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 3, min_samples_leaf = 5)

    adb_clf = AdaBoostClassifier(tree_clf, algorithm="SAMME", n_estimators=200)
    adb_clf.fit(X_train, Y_train)

    scores = cross_val_score(adb_clf, X_test, Y_test, cv = 5)
    print(scores.mean())
    return adb_clf

# Function to make predictions 
def prediction(X_test, clf): 
  
    # Predicton on test with giniIndex 
    y_pred = clf.predict(X_test) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", metrics.accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", metrics.classification_report(y_test, y_pred)) 


if __name__ == "__main__":
    X_tr, X_te, Y_tr, Y_te = load_data()
    clf = train(X_tr, X_te, Y_tr, Y_te)
    y_pred = prediction(X_te, clf) 
    cal_accuracy(Y_te, y_pred) 
