from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd

# Function to perform training with giniIndex. 
def train_using_gini(X_train, y_train): 
  
    # Creating the classifier object 
    clf_gini = tree.DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = tree.DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 


# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", metrics.accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", metrics.classification_report(y_test, y_pred)) 


def main(): 
      
    # Building Phase 

    Dataset = pd.read_csv("../train.csv", sep = ',', header = None)

    # split the dataset
    # seperating the label and features 
    X = Dataset.values[1:, 2:]
    Y = Dataset.values[1:, 1]

    #X = np.ones([5, 5])
    #Y = np.ones([5,])

    size = X.shape
    X = np.asarray(X).astype(float)
    Y = np.reshape(Y, [size[0]]).astype(int)

    # normalize
    X_mean = np.mean(X, axis=0)
    for i in range(size[1]):
        X[:, i] = X[:, i]/X_mean[i]
   



    # Spliting the dataset into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

    clf_gini = train_using_gini(X_train, Y_train) 
    clf_entropy = tarin_using_entropy(X_train, Y_train) 
    #print(Y_train.describe())

    print(np.sum(Y_test))



    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(Y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(Y_test, y_pred_entropy) 
      
      
if __name__=="__main__": 
    main() 
