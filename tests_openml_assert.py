import openml as openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from LDAClassifier import LDAClassifier

datasets_id = [11, 37, 41, 54, 61, 187, 329, 464, 468, 472, \
               683, 694, 874, 894, 969, 973, 974, 994, 997, 1005, \
               1015, 1048, 1060, 1061, 1062, 1063, 1064, 1073, 1075, 1117]
for id in datasets_id:
    dataset = openml.datasets.get_dataset(id)
    (X, y, categorical, names) = dataset.get_data(target=dataset.default_target_attribute)
    X = np.array(X)
    if id == 1048 or id == 1073 or id == 1060 or id == 1061 or id == 1062 or id == 1064:
        y = y.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5656)
    # training and predicting of online LDA
    classifier_online = LDAClassifier(type="online", regularization_coefficient=0.02).fit(X_train[:, 0], y_train)
    predicted_training_online = classifier_online.predict(X_train[:, 0])
    for i in range(1, X_train.shape[1]):
        classifier_online = classifier_online.add_feature(X_train[:, i])
        predicted_training_online = classifier_online.predict(X_train[:, :i + 1])
    predicted_testing_online = classifier_online.predict(X_test)
    # training and predicting of offline LDA
    classifier_offline = LDAClassifier(type="offline", regularization_coefficient=0.02)
    predicted_training_offline = []
    for i in range(0, X_train.shape[1]):
        classifier_offline = classifier_offline.fit(X_train[:, :i + 1], y_train)
        predicted_training_offline = classifier_offline.predict(X_train[:, :i + 1])
    predicted_testing_offline = classifier_offline.predict(X_test)
    # training and predicting of sklearn LDA
    classifier_sklearn = LDA(solver='lsqr', shrinkage=0.02)
    predicted_training_sklearn = []
    for i in range(0, X_train.shape[1]):
        classifier_sklearn = classifier_sklearn.fit(X_train[:, :i + 1], y_train)
        predicted_training_sklearn = classifier_sklearn.predict(X_train[:, :i + 1])
    predicted_testing_sklearn = classifier_sklearn.predict(X_test)
    # comparing results of prediction
    np.testing.assert_equal(predicted_training_online, predicted_training_offline)
    np.testing.assert_equal(predicted_training_online, predicted_training_sklearn)
    np.testing.assert_equal(predicted_testing_online, predicted_testing_offline)
    np.testing.assert_equal(predicted_testing_online, predicted_testing_sklearn)
    # comparing posterior probabilities
    np.testing.assert_equal(classifier_online.prob_classes, classifier_offline.prob_classes)
    np.testing.assert_equal(classifier_online.prob_classes, classifier_sklearn.priors_)
    # comparing total covariance matrix
    np.testing.assert_almost_equal(classifier_online.total_covariance(), classifier_offline.total_covariance())
    np.testing.assert_almost_equal(classifier_online.total_covariance(), classifier_sklearn.covariance_)
    np.testing.assert_almost_equal(classifier_offline.total_covariance(), classifier_sklearn.covariance_)
    # comparing means
    np.testing.assert_almost_equal(classifier_online.means_by_class(), classifier_offline.means_by_class())
    np.testing.assert_almost_equal(classifier_online.means_by_class(), classifier_sklearn.means_)

    # simplify due to specific of sklearn
    if len(np.unique(y)) > 2:
        np.testing.assert_almost_equal(classifier_online.coef(), classifier_offline.coef())
        np.testing.assert_almost_equal(classifier_online.coef(), classifier_sklearn.coef_)
        np.testing.assert_almost_equal(classifier_online.intercept(), classifier_offline.intercept())
        np.testing.assert_almost_equal(classifier_online.intercept(), classifier_sklearn.intercept_, decimal=5)
        np.testing.assert_almost_equal(classifier_online.scores, classifier_offline.scores)
        np.testing.assert_almost_equal(classifier_online.scores, classifier_sklearn.decision_function(X_test), decimal=5)






