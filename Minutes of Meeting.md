#EC503 Minutes of Meeting

Firstly, we use several algorithms and try to classify the data.
We used DRO, Logistic Regression, KNN, SVM, Decision Tree. 🌲 
And we found that the result was bad.
Althouth the accuracy were high for all algorithms, that was because the dataset has large bias: 90% of the data are labeled as 0, only 10% of data are labeled as 1. Thus, the accuracy will be 90% even if the classifier predict all test data as 0.
And the AUC was significantly low: only 0.5
Even if we normalize the data, the results had nothing change: still very bad.




Plan for the second week:
1. 把label为1的数据提取出来，让label为1的数据的比例大一些，然后看一下结果是否有好转
2. 想办法检测outlier并去除：identify outlier
3. 需要降维吗？
4. 能否试一下：不以整体精确度为导向，而是以“把1都预测对，然后再考虑0”为导向
5. 看一下Kaggle里面的kernel https://www.kaggle.com/c/santander-customer-transaction-prediction/kernels
6. 查一下对数据有bia的情况下的检测方法



2019.4.6 15:00 yzsu
重新分配了数据label的密度，使得label为1的data与label为0的data的比例为1:1（原本是1:9），然后再进行测量，得到的结果是：
ROC AUC:0.788
Accuracy:0.787
需要注意的是，我使用的SVM是sklearn里的sklearn.svm.SVC(),默认是使用了kernel的，使用的是‘rbf’即Gaussian Kernel。
然后试了一下使用Polynomial Kernel，默认的Degree = 3，得到的结果是：
ROC AUC:0.780
Accuracy:0.779
