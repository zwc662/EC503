#EC503 Minutes of Meeting
Firstly, we use several algorithms and try to classify the data.
We used DRO, Logistic Regression, KNN, SVM, Decision Tree. 🌲 
And we found that the result was bad.
Althouth the accuracy were high for all algorithms, that was because the dataset has large bias: 90% of the data are labeled as 0, only 10% of data are labeled as 1. Thus, the accuracy will be 90% even if the classifier predict all test data as 0.
And the AUC was significantly low: only 0.5
Even if we normalize the data, the results had nothing change: still very bad.
