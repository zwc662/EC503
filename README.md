# EC503

20190331
====

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


Suggestions:

* Survey on the problem of bias in dataset.
  * <a href="http://undoingbias.csail.mit.edu">Undoing the Damage of Dataset Bias</a></ol
  * <a href="http://people.csail.mit.edu/torralba/research/bias/">Unbiased Look at Dataset Bias</a></ol>


* Survey on outlier detection problems
  * <a href="https://arxiv.org/pdf/1401.6424.pdf">Toward Supervised Anomaly Detection</a>
  * <a href="https://dl.acm.org/citation.cfm?id=1263334">K-Means+ID3: A/ Novel Method for Supervised Anomaly Detection by Cascading K-Means Clustering and ID3 Decision Tree Learning Methods</a>
  * <a href="https://dl.acm.org/citation.cfm?id=1541882">Anomaly detection: A survey</a>

* <a href="https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html#from-random-over-sampling-to-smote-and-adasyn">Imbalanced Learn</a>
