# Linear Discriminant Analysis on Stream of Features

Python implementation of LDA classifier expanded with **feature incrementation**. When a new feature is added to the model, the internal structure is being updated instead of retraing the model from the beginning. Multicollinearity is handled with regularization using a constant input regularization parameter. An inverse of a total covariance matrix is solved with the Cholesky decomposition. The classifier operates in two modes: "offline" means traditional batch processing, when the data comes in bulk, "online" stands for feature incremental learning. 
## Example of usage 
```python
dataset = load_iris(True)
X = dataset[0]
y = dataset[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5656)
classifier = LDAClassifier(type="offline", regularization_coefficient=0.02)
classifier.fit(self.X_train[:, 0], y_train)
for i in range(1, self.X_train.shape[1]):
    classifier.add_feature(X_train[:, i])
classifier.predict(X_test)
```

## Brief description of files

* *LDAClassifier.py* main file 
* *tests_openml.py* comparing performance of LDA-online to LDA-offline and [LDA-sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis) using different metrics and 30 OpenML datasets
* *tests_openml_assert.py* comparing the internal structure of LDA-online to LDA-offline and [LDA-sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
* *tests_synt.py* comparing performance of LDA-online to LDA-offline and [LDA-sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis) using synthetic data 

**Do not forget** to change [shrunk_covariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.shrunk_covariance.html), which is used by [LDA-sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis), as follows to get comparative results. 

```python
shrunk_cov.flat[::n_features + 1] += shrinkage 
```
