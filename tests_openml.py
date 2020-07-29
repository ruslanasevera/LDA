import time
import openml as openml
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from LDAClassifier import LDAClassifier
from numpy import unravel_index

pd.set_option('display.max_columns', 15)

def exp_normalize(x):
    #np.seterr(all='ignore')
    b = x.max()
    y = np.exp(x - b)
    return y / np.array(y).sum()

def exp_naive(x):
    #np.seterr(all='ignore')
    y = np.exp(x)
    return y / np.array(y).sum()

# getting prior probabilities from results of the discriminant function
def score_to_prob(scores):
    #np.seterr(all='ignore')
    scores = np.array(scores)
    probs = [exp_normalize(i) for i in scores]
    return np.array(probs)

def brier_multi(targets, probs, classes):
    np.seterr(all='ignore')
    targets = np.array(targets)
    probs = np.array(probs)
    for i in range(len(probs)):
        probs[i,classes.index(targets[i])] -= 1
    return np.mean(np.sum(probs**2, axis=1))

def offline_training_predicting_time(model, X_train, y_train, X_test, times=10):
    fit_time_array = []
    predict_time_array = []
    predicted = []
    for i in range(times):
        fit_time = time.process_time()
        model = model.fit(X_train, y_train)
        fit_time = time.process_time() - fit_time
        predict_time = time.process_time()
        predicted = model.predict(X_test)
        predict_time = time.process_time() - predict_time
        fit_time_array.append(fit_time)
        predict_time_array.append(predict_time)
    return model, predicted, np.mean(fit_time_array), np.mean(predict_time_array), np.std(fit_time_array), np.std(predict_time_array)

def online_training_predicting_time(model, X_train, y_train, X_test, times=10, add_feature=False):
    fit_time_array = []
    predict_time_array = []
    predicted_testing = []
    predicted_training = []
    for i in range(times):
        fit_time = 0
        predict_time = 0
        if add_feature:
            tmp = time.process_time()
            model = model.fit(X_train[:, 0], y_train)
            fit_time += time.process_time() - tmp
            tmp = time.process_time()
            predicted_training = model.predict(X_train[:, 0])
            predict_time += time.process_time() - tmp
            for i in range(1, X_train.shape[1]):
                tmp = time.process_time()
                model = model.add_feature(X_train[:, i])
                fit_time += time.process_time() - tmp
                tmp = time.process_time()
                predicted_training = model.predict(X_train[:, :i + 1])
                predict_time += time.process_time() - tmp
        else:
            for i in range(0, X_train.shape[1]):
                tmp = time.process_time()
                model = model.fit(X_train[:,:i+1], y_train)
                fit_time += time.process_time() - tmp
                tmp = time.process_time()
                predicted_training = model.predict(X_train[:,:i+1])
                predict_time += time.process_time() - tmp
        fit_time_array.append(fit_time)
        predict_time_array.append(predict_time)
    predicted_testing = model.predict(X_test)
    return model, predicted_training, predicted_testing, np.mean(fit_time_array), np.mean(predict_time_array), np.std(fit_time_array), np.std(predict_time_array)

classifiers = ['sklearn', 'my_offline', 'my_online']

openml_list = openml.datasets.list_datasets()
datalist = pd.DataFrame.from_dict(openml_list, orient="index")
datalist = datalist[['did', 'name', 'NumberOfInstances',
                     'NumberOfFeatures', 'NumberOfClasses', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumberOfInstancesWithMissingValues', 'uploader', 'MinorityClassSize']]
filtered = datalist.query('NumberOfClasses > 1')
filtered = filtered.query('NumberOfClasses <= 10')
filtered = filtered.query('NumberOfInstances < 1000')
filtered = filtered.query('NumberOfInstances > 20')
filtered = filtered.query('NumberOfFeatures <= 30')
filtered = filtered.query('NumberOfInstancesWithMissingValues == 0')
filtered = filtered.query('NumberOfSymbolicFeatures == 1')
filtered = filtered.query('uploader in ["1", "2"]')
filtered = filtered.query('MinorityClassSize > 2')

print(len(filtered))
successful_datasets = 0
df_datasets = pd.DataFrame(columns=['ID','Name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses', 'Max. Pearson corr. coef.'])
df_cohen_kappa = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])
df_f1 = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])
df_recall = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])
df_roc_auc = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])
df_brier = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])
df_training_precision = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])
df_testing_precision = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])
df_elapsed_time_train = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])
df_elapsed_time_predict = pd.DataFrame(columns=['ID','Name', 'sklearn', 'my_offline', 'my_online'])

i = 0
for row in filtered.iterrows():
    try:
        name = row[1]['name']
        dataset = openml.datasets.get_dataset(name)
        id = row[0]
        number_of_instances = row[1]['NumberOfInstances']
        number_of_features = row[1]['NumberOfFeatures']
        number_of_classes = row[1]['NumberOfClasses']
        (X, y, categorical, names) = dataset.get_data(target=dataset.default_target_attribute)
        X = np.array(X)
        if ("jEdit_" in name) or ('ar' in name):
            y = y.astype('int64')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5656)
        if y.ndim != 1 or y_train.ndim != 1 or y_test.ndim != 1:
            raise Exception('y is not 1d')
        for cl in np.unique(y):
            if (X_train[y_train==cl].shape[0] < 2) or (X_test[y_test==cl].shape[0] < 2):
                raise Exception('dataset ' + name + ' has too little samples: ' + str(X[y==cl].shape[0]) + ' for class: ' + str(cl))
        cohen_kappa_array = []
        f1_array = []
        recall_array = []
        training_precision_array = []
        testing_precision_array = []
        roc_auc_array = []
        brier_array = []
        elapsed_time_train_array = []
        elapsed_time_predict_array = []
        try:
            for k in range(len(classifiers)):
                print('trying: ', name, ' on ', classifiers[k])
                cohen_kappa = 'cohen_kappa?'
                f1 = 'f1?'
                recall = 'recall?'
                training_precision = 'training_precision?'
                testing_precision = 'testing_precision?'
                roc_auc = 'roc_auc?'
                brier = 'brier?'
                elapsed_time_train = 'elapsed_time_train?'
                elapsed_time_predict = 'elapsed_time_predict?'
                classifier = 0
                predicted_training = []
                predicted_testing = []
                elapsed_time_train_std = 0
                elapsed_time_predict_std = 0
                if classifiers[k] == "sklearn":
                    classifier = LDA(solver='lsqr', shrinkage=0.02)
                    classifier, predicted_training, predicted_testing, elapsed_time_train, elapsed_time_predict, elapsed_time_train_std, elapsed_time_predict_std = \
                        online_training_predicting_time(classifier, X_train, y_train, X_test)
                elif classifiers[k] == "my_offline":
                    classifier = LDAClassifier(type="offline",
                                               regularization_coefficient=0.02)
                    classifier, predicted_training, predicted_testing, elapsed_time_train, elapsed_time_predict, elapsed_time_train_std, elapsed_time_predict_std = \
                        online_training_predicting_time(classifier, X_train, y_train, X_test)
                elif classifiers[k] == "my_online":
                    classifier = LDAClassifier(type="online",
                                               regularization_coefficient=0.02)
                    classifier, predicted_training, predicted_testing, elapsed_time_train, elapsed_time_predict, elapsed_time_train_std, elapsed_time_predict_std = \
                        online_training_predicting_time(classifier, X_train, y_train, X_test, add_feature=True)
                cohen_kappa = cohen_kappa_score(predicted_testing, y_test)
                cohen_kappa_array.append(cohen_kappa)
                f1 = f1_score(predicted_testing, y_test, average='micro')
                f1_array.append(f1)
                recall = recall_score(predicted_testing, y_test, average='micro')
                recall_array.append(recall)
                testing_precision = precision_score(predicted_testing, y_test, average='micro')
                testing_precision_array.append(testing_precision)
                training_precision = precision_score(predicted_training, y_train, average='micro')
                training_precision_array.append(training_precision)
                elapsed_time_predict_array.append(elapsed_time_predict)
                elapsed_time_train_array.append(elapsed_time_train)
                predict_proba = 0
                class_labels = 0
                if classifiers[k] == "sklearn":
                    predict_proba = classifier.predict_proba(X_test)
                    class_labels = classifier.classes_
                else:
                    predict_proba = np.array(score_to_prob(classifier.scores))
                    class_labels = classifier.class_labels
                predict_proba_for_roc_auc = []
                if len(class_labels) <= 2:
                    predict_proba_for_roc_auc = np.max(predict_proba, axis=1)
                else:
                    predict_proba_for_roc_auc = predict_proba/predict_proba.sum(axis=1,keepdims=1)
                brier = brier_multi(y_test, predict_proba, list(class_labels))
                brier_array.append(brier)
                if len(class_labels) <= 2:
                    roc_auc = roc_auc_score(y_true=y_test, y_score=predict_proba_for_roc_auc, average='macro')
                else:
                    roc_auc = roc_auc_score(y_true=y_test, y_score=predict_proba_for_roc_auc, average='macro', multi_class='ovo', labels=class_labels)
                roc_auc_array.append(roc_auc)
                if k == len(classifiers)-1:
                    cor = np.corrcoef(X, rowvar=False)
                    cor = np.abs(cor)
                    cor_without_diagonal = cor[~np.eye(cor.shape[0],dtype=bool)].reshape(cor.shape[0],-1)
                    max_index = unravel_index(np.argmax(cor_without_diagonal, axis=None), cor_without_diagonal.shape)
                    if number_of_features != X.shape[1]:
                        print("number of features error: ", id, name, number_of_features, X.shape[1])
                        number_of_features = X.shape[1]
                    if number_of_classes != len(np.unique(y)):
                        print("number of classes error: ", id, name, number_of_classes, len(np.unique(y)), np.unique(y))
                        number_of_classes = len(np.unique(y))
                    if number_of_instances != X.shape[0]:
                        print("number of instances error: ", id, name, number_of_instances, X.shape[0])
                        number_of_instances = X.shape[0]
                    df_datasets.loc[i] = [id, name, int(number_of_instances), int(number_of_features), int(number_of_classes), round(cor_without_diagonal[max_index], 3)]
                    df_cohen_kappa.loc[i] = [id, name, round(cohen_kappa_array[0], 5), round(cohen_kappa_array[1], 5), round(cohen_kappa_array[2], 5)]
                    df_f1.loc[i] = [id, name, round(f1_array[0], 5), round(f1_array[1], 5), round(f1_array[2], 5)]
                    df_recall.loc[i] = [id, name, round(recall_array[0], 5), round(recall_array[1], 5), round(recall_array[2], 5)]
                    df_training_precision.loc[i] = [id, name, round(training_precision_array[0], 5), round(training_precision_array[1], 5), round(training_precision_array[2], 5)]
                    df_testing_precision.loc[i] = [id, name, round(testing_precision_array[0], 5), round(testing_precision_array[1], 5), round(testing_precision_array[2], 5)]
                    df_roc_auc.loc[i] = [id, name, round(roc_auc_array[0], 5), round(roc_auc_array[1], 5), round(roc_auc_array[2], 5)]
                    df_brier.loc[i] = [id, name, round(brier_array[0], 5), round(brier_array[1], 5), round(brier_array[2], 5)]
                    df_elapsed_time_predict.loc[i] = [id, name, round(elapsed_time_predict_array[0], 5), round(elapsed_time_predict_array[1], 5), round(elapsed_time_predict_array[2], 5)]
                    df_elapsed_time_train.loc[i] = [id, name, round(elapsed_time_train_array[0], 5), round(elapsed_time_train_array[1], 5), round(elapsed_time_train_array[2], 5)]
                    # metrics measurements without rounding
                    '''df_cohen_kappa.loc[i] = [id, name, cohen_kappa_array[0], cohen_kappa_array[1], cohen_kappa_array[2]]
                    df_f1.loc[i] = [id, name, f1_array[0], f1_array[1], f1_array[2]]
                    df_recall.loc[i] = [id, name, recall_array[0], recall_array[1],recall_array[2]]
                    df_training_precision.loc[i] = [id, name, training_precision_array[0],training_precision_array[1],training_precision_array[2]]
                    df_testing_precision.loc[i] = [id, name, testing_precision_array[0],testing_precision_array[1],testing_precision_array[2]]
                    df_roc_auc.loc[i] = [id, name, roc_auc_array[0], roc_auc_array[1],roc_auc_array[2]]
                    df_brier.loc[i] = [id, name, brier_array[0], brier_array[1],brier_array[2]]
                    df_elapsed_time_predict.loc[i] = [id, name, elapsed_time_predict_array[0],elapsed_time_predict_array[1],elapsed_time_predict_array[2]]
                    df_elapsed_time_train.loc[i] = [id, name, elapsed_time_train_array[0],elapsed_time_train_array[1],elapsed_time_train_array[2]]'''
                    i += 1
                    successful_datasets += 1
                    # metrics measurements without rounding
                    '''if not (cohen_kappa_array[0] == cohen_kappa_array[1] and  cohen_kappa_array[0] == cohen_kappa_array[2]):
                        print('not equal:', id, name, cohen_kappa_array[0], cohen_kappa_array[1], cohen_kappa_array[2])
                    if not (f1_array[0] == f1_array[1] and f1_array[0] == f1_array[2]):
                        print('not equal:', id, name, f1_array[0], f1_array[1], f1_array[2])
                    if not (recall_array[0] == recall_array[1] and recall_array[0] == recall_array[2]):
                        print('not equal:', id, name, recall_array[0], recall_array[1], recall_array[2])
                    if not (training_precision_array[0] == training_precision_array[1] and training_precision_array[0] == training_precision_array[2]):
                        print('not equal:', id, name, training_precision_array[0], training_precision_array[1], training_precision_array[2])
                    if not (testing_precision_array[0] == testing_precision_array[1] and testing_precision_array[0] == testing_precision_array[2]):
                        print('not equal:', id, name, testing_precision_array[0], testing_precision_array[1], testing_precision_array[2])
                    if not (roc_auc_array[0] == roc_auc_array[1] and roc_auc_array[0] == roc_auc_array[2]):
                        print('not equal:', id, name, roc_auc_array[0], roc_auc_array[1], roc_auc_array[2])
                    if not (brier_array[0] == brier_array[1] and brier_array[0] == brier_array[2]):
                        print('not equal:', id, name, brier_array[0], brier_array[1], brier_array[2])'''
                    if not (round(cohen_kappa_array[0], 7) == round(cohen_kappa_array[1], 7) and round(cohen_kappa_array[0], 7) == round(cohen_kappa_array[2], 7)):
                        print('not equal:', id, name, cohen_kappa_array[0], cohen_kappa_array[1], cohen_kappa_array[2])
                    if not (round(f1_array[0], 7) == round(f1_array[1], 7) and round(f1_array[0], 7) == round(f1_array[2], 7)):
                        print('not equal:', id, name, f1_array[0], f1_array[1], f1_array[2])
                    if not (round(recall_array[0], 7) == round(recall_array[1], 7) and round(recall_array[0], 7) == round(recall_array[2], 7)):
                        print('not equal:', id, name, recall_array[0], recall_array[1], recall_array[2])
                    if not (round(training_precision_array[0], 7) == round(training_precision_array[1], 7) and round(training_precision_array[0], 7) == round(training_precision_array[2], 7)):
                        print('not equal:', id, name, training_precision_array[0], training_precision_array[1],
                              training_precision_array[2])
                    if not (round(testing_precision_array[0], 7) == round(testing_precision_array[1], 7) and round(testing_precision_array[0], 7) == round(testing_precision_array[2], 7)):
                        print('not equal:', id, name, testing_precision_array[0], testing_precision_array[1], testing_precision_array[2])
                    if not (round(roc_auc_array[0], 7) == round(roc_auc_array[1], 7) and round(roc_auc_array[0], 7) == round(roc_auc_array[2], 7)):
                        print('not equal:', id, name, roc_auc_array[0], roc_auc_array[1], roc_auc_array[2])
                    if not (round(brier_array[0], 7) == round(brier_array[1], 7) and round(brier_array[0], 7) == round(brier_array[2], 7)):
                        print('not equal:', id, name, brier_array[0], brier_array[1], brier_array[2])

        except Exception as e:
            print('dataset ' +  name + ' is not suitable for LDA: ', e)
            print(name, cohen_kappa, f1, recall, training_precision, testing_precision, roc_auc, brier, elapsed_time_train, elapsed_time_predict)
    except Exception as e:
        print(e)

print('successful datasets: ', i)
print(df_datasets.head(30))
df_datasets.to_csv('datasets.csv', index=False)
print('cohen_kappa')
print(df_cohen_kappa.head(30))
df_cohen_kappa.to_csv('cohen_kappa.csv', index=False)
print('recall')
print(df_recall.head(30))
df_recall.to_csv('recall.csv', index=False)
print('f1')
print(df_f1.head(30))
df_f1.to_csv('f1.csv', index=False)
print('training_precision')
print(df_training_precision.head(30))
df_training_precision.to_csv('training_precision.csv', index=False)
print('testing_precision')
print(df_testing_precision.head(30))
df_testing_precision.to_csv('testing_precision.csv', index=False)
print('roc_auc')
print(df_roc_auc.head(30))
df_roc_auc.to_csv('roc_auc.csv', index=False)
print('brier')
print(df_brier.head(30))
df_brier.to_csv('brier.csv', index=False)
print('elapsed_time_train')
print(df_elapsed_time_train.head(30))
df_elapsed_time_train.to_csv('elapsed_time_train.csv', index=False)
print('elapsed_time_predict')
print(df_elapsed_time_predict.head(30))
df_elapsed_time_predict.to_csv('elapsed_time_predict.csv', index=False)
