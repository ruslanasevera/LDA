import time
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from LDAClassifier import LDAClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

pd.set_option('display.max_columns', 15)

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def exp_naive(x):
    y = np.exp(x)
    return y / y.sum()

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

# return online training time
def online_training_predicting_time(model, X_train, y_train, X_test, times=5, add_feature=False):
    fit_time_array = []
    predict_time_array = []
    predicted_testing = []
    predicted_training = []
    fit_time = 0
    predict_time = 0
    if add_feature:
        tmp = time.process_time()
        model = model.fit(X_train[:, 0], y_train)
        fit_time += time.process_time() - tmp
        fit_time_array.append(fit_time)
        '''predict_time_array_inside = []
        for j in range(times):
            tmp = time.process_time()
            predicted_training = model.predict(X_train[:, 0])
            predict_time_array_inside.append(time.process_time() - tmp)
        predict_time += np.mean(predict_time_array_inside)
        predict_time_array.append(predict_time_array)'''
        for i in range(1, X_train.shape[1]):
            tmp = time.process_time()
            model = model.add_feature(X_train[:, i])
            fit_time += time.process_time() - tmp
            fit_time_array.append(fit_time)
            '''predict_time_array_inside = []
            for j in range(times):
                tmp = time.process_time()
                predicted_training = model.predict(X_train[:, :i + 1])
                predict_time_array_inside.append(time.process_time() - tmp)
            predict_time += np.mean(predict_time_array_inside)
            predict_time_array.append(predict_time_array)'''
    else:
        for i in range(0, X_train.shape[1]):
            tmp = time.process_time()
            model = model.fit(X_train[:,:i+1], y_train)
            fit_time += time.process_time() - tmp
            fit_time_array.append(fit_time)
            '''predict_time_array_inside = []
            for j in range(times):
                tmp = time.process_time()
                predicted_training = model.predict(X_train[:, :i + 1])
                predict_time_array_inside.append(time.process_time() - tmp)
            predict_time += np.mean(predict_time_array_inside)
            predict_time_array.append(predict_time_array)'''
    #predicted_testing = model.predict(X_test)
    return model, predicted_training, predicted_testing, np.array(fit_time_array), np.array(predict_time_array)

# return arrays for graphs
def online_training_predicting_time_2(model, X_train, y_train, X_test, times=5, add_feature=False):
    predicted_testing = []
    predicted_training = []
    fit_time = 0
    predict_time = 0
    if add_feature:
        tmp = time.process_time()
        model = model.fit(X_train[:, 0], y_train)
        fit_time += time.process_time() - tmp
        #tmp = time.process_time()
        #predicted_training = model.predict(X_train[:, 0])
        #predict_time += time.process_time() - tmp
        for i in range(1, X_train.shape[1]):
            tmp = time.process_time()
            model = model.add_feature(X_train[:, i])
            fit_time += time.process_time() - tmp
            #tmp = time.process_time()
            #predicted_training = model.predict(X_train[:, :i + 1])
            #predict_time += time.process_time() - tmp
    else:
        for i in range(0, X_train.shape[1]):
            tmp = time.process_time()
            model = model.fit(X_train[:,:i+1], y_train)
            fit_time += time.process_time() - tmp
            #tmp = time.process_time()
            #predicted_training = model.predict(X_train[:,:i+1])
            #predict_time += time.process_time() - tmp
    #predicted_testing = model.predict(X_test)
    return model, predicted_training, predicted_testing, fit_time, predict_time

# return online predicting time
def online_predicting_time(model, X_train, y_train, X_test, times=5, add_feature=False):
    predicted_testing = []
    predicted_training = []
    fit_time = 0
    predict_time_array = []
    model = model.fit(X_train, y_train)
    for i in range(times):
        tmp = time.process_time()
        predicted_testing = model.predict(X_test)
        predict_time = time.process_time() - tmp
        predict_time_array.append(predict_time)
    return model, predicted_training, predicted_testing, fit_time, np.mean(predict_time_array)

classifiers = ['sklearn', 'my_offline', 'my_online']

#dataset = make_classification(n_samples=300, n_features=30, n_informative=20, n_redundant=5, n_repeated=3, n_classes=5)
#dataset_ = make_classification(n_samples=300, n_features=50, n_informative=40, n_redundant=5, n_repeated=3, n_classes=15)
dataset = make_classification(n_samples=2000, n_features=500, n_informative=400, n_redundant=25, n_repeated=25, n_classes=15)
#dataset = make_classification(n_samples=3000, n_features=2000, n_informative=1500, n_redundant=300, n_repeated=100, n_classes=15)
X = np.array(dataset[0])
y = np.array(dataset[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5656)
elapsed_time_predict_online_array = []
elapsed_time_predict_offline_array = []
elapsed_time_predict_sklearn_array = []
elapsed_time_train_online_array = []
elapsed_time_train_offline_array = []
elapsed_time_train_sklearn_array = []
predicted_training_sklearn, predicted_testing_sklearn = 0, 0
predicted_training_offline, predicted_testing_offline = 0, 0
predicted_training_online, predicted_testing_online = 0, 0
classifier_offline = LDAClassifier(type="offline", regularization_coefficient=0.02)
classifier_online = LDAClassifier(type="online", regularization_coefficient=0.02, max_num_of_features=2000)
classifier_sklearn = LDA(solver='lsqr', shrinkage=0.02)
# for testing training acceleration
classifier_sklearn, predicted_training_sklearn, predicted_testing_sklearn, elapsed_time_train_sklearn, elapsed_time_predict_sklearn = \
    online_training_predicting_time(classifier_sklearn, X_train, y_train, X_test)
classifier_offline, predicted_training_offline, predicted_testing_offline, elapsed_time_train_offline, elapsed_time_predict_offline = \
    online_training_predicting_time(classifier_offline, X_train, y_train, X_test)
classifier_online, predicted_training_online, predicted_testing_online, elapsed_time_train_online, elapsed_time_predict_online = \
    online_training_predicting_time(classifier_online, X_train, y_train, X_test, add_feature=True)
print('my fit last:', elapsed_time_train_offline[X_train.shape[1]-1]/elapsed_time_train_online[X_train.shape[1]-1])
print('sklearn fit last:', elapsed_time_train_sklearn[X_train.shape[1]-1]/elapsed_time_train_online[X_train.shape[1]-1])
print('my fit:', np.mean(np.array(elapsed_time_train_offline)/np.array(elapsed_time_train_online)))
print('sklearn fit:', np.mean(np.array(elapsed_time_train_sklearn)/np.array(elapsed_time_train_online)))
# for describing dependence on the amount of features
'''classifier_sklearn, predicted_training_sklearn, predicted_testing_sklearn, elapsed_time_train_sklearn, elapsed_time_predict_sklearn = \
    online_training_predicting_time(classifier_sklearn, X_train, y_train, X_test)
classifier_offline, predicted_training_offline, predicted_testing_offline, elapsed_time_train_offline, elapsed_time_predict_offline = \
    online_training_predicting_time(classifier_offline, X_train, y_train, X_test)
classifier_online, predicted_training_online, predicted_testing_online, elapsed_time_train_online, elapsed_time_predict_online = \
    online_training_predicting_time(classifier_online, X_train, y_train, X_test, add_feature=True)
elapsed_time_predict_online_array = elapsed_time_predict_online
elapsed_time_predict_offline_array = elapsed_time_predict_offline
elapsed_time_predict_sklearn_array = elapsed_time_predict_sklearn
elapsed_time_train_online_array = elapsed_time_train_online
elapsed_time_train_offline_array = elapsed_time_train_offline
elapsed_time_train_sklearn_array = elapsed_time_train_sklearn'''
# for describing dependence on the amount of samples
'''for i in range(700, 2001, 100):
    classifier_sklearn, predicted_training_sklearn, predicted_testing_sklearn, elapsed_time_train_sklearn, elapsed_time_predict_sklearn = \
        online_training_predicting_time_2(classifier_sklearn, X_train[:i+1, :], y_train[:i+1], X_test[:i+1, :])
    classifier_offline, predicted_training_offline, predicted_testing_offline, elapsed_time_train_offline, elapsed_time_predict_offline = \
        online_training_predicting_time_2(classifier_offline, X_train[:i+1, :], y_train[:i+1], X_test[:i+1, :])
    classifier_online, predicted_training_online, predicted_testing_online, elapsed_time_train_online, elapsed_time_predict_online = \
        online_training_predicting_time_2(classifier_online, X_train[:i+1, :], y_train[:i+1], X_test[:i+1, :], add_feature=True)
    elapsed_time_predict_online_array.append(elapsed_time_predict_online)
    elapsed_time_predict_offline_array.append(elapsed_time_predict_offline)
    elapsed_time_predict_sklearn_array.append(elapsed_time_predict_sklearn)
    elapsed_time_train_online_array.append(elapsed_time_train_online)
    elapsed_time_train_offline_array.append(elapsed_time_train_offline)
    elapsed_time_train_sklearn_array.append(elapsed_time_train_sklearn)'''

# for testing metrics
'''df = pd.DataFrame(columns=['Name', 'sklearn', 'my_offline', 'my_online'])
df.loc[0] = ['cohen_kappa', cohen_kappa_score(predicted_testing_sklearn, y_test), cohen_kappa_score(predicted_testing_offline, y_test), cohen_kappa_score(predicted_testing_online, y_test)]
df.loc[1] = ['f1_score', f1_score(predicted_testing_sklearn, y_test, average='micro'), f1_score(predicted_testing_offline, y_test, average='micro'), f1_score(predicted_testing_online, y_test, average='micro')]
df.loc[2] = ['recall', recall_score(predicted_testing_sklearn, y_test, average='micro'), recall_score(predicted_testing_offline, y_test, average='micro'), recall_score(predicted_testing_online, y_test, average='micro')]
df.loc[3] = ['precision_score_testing', precision_score(predicted_testing_sklearn, y_test, average='micro'), precision_score(predicted_testing_offline, y_test, average='micro'), precision_score(predicted_testing_online, y_test, average='micro')]
df.loc[4] = ['precision_score_training', precision_score(predicted_training_sklearn, y_train, average='micro'), precision_score(predicted_training_offline, y_train, average='micro'), precision_score(predicted_training_online, y_train, average='micro')]
predict_proba_sklearn = classifier_sklearn.predict_proba(X_test)
class_labels_sklearn = classifier_sklearn.classes_
classifier_offline.predict(X_test)
predict_proba_offline = score_to_prob(classifier_offline.scores)
class_labels_offline = classifier_offline.class_labels
classifier_online.predict(X_test)
predict_proba_online = score_to_prob(classifier_online.scores)
class_labels_online = classifier_online.class_labels
predict_proba_for_roc_auc_sklearn = predict_proba_sklearn/predict_proba_sklearn.sum(axis=1,keepdims=1)
predict_proba_for_roc_auc_offline = predict_proba_offline/predict_proba_offline.sum(axis=1,keepdims=1)
predict_proba_for_roc_auc_online = predict_proba_online/predict_proba_online.sum(axis=1,keepdims=1)
df.loc[5] = ['brier', brier_multi(y_test, predict_proba_sklearn, list(class_labels_sklearn)), brier_multi(y_test, predict_proba_offline, list(class_labels_offline)), brier_multi(y_test, predict_proba_online, list(class_labels_online))]
df.loc[6] = ['roc_auc', roc_auc_score(y_true=y_test, y_score=predict_proba_for_roc_auc_sklearn, average='macro', multi_class='ovo', labels=class_labels_sklearn), \
             roc_auc_score(y_true=y_test, y_score=predict_proba_for_roc_auc_offline, average='macro', multi_class='ovo', labels=class_labels_offline), \
             roc_auc_score(y_true=y_test, y_score=predict_proba_for_roc_auc_online, average='macro', multi_class='ovo', labels=class_labels_online)]
print(df.head(10))'''
#numbers = [i for i in range(700, 2001, 100)]
#numbers = [i for i in range(1, 501)]
# for graphs
# do not forget to change names fig.savefig('name.pdf')
'''fig = plt.figure()
ax = fig.add_subplot()
ax.plot(numbers, elapsed_time_predict_sklearn_array, color='green', label='scikit-learn')
ax.plot(numbers, elapsed_time_predict_offline_array, color='orange', label='offline')
ax.plot(numbers, elapsed_time_predict_online_array, color='pink', label='online')
ax.set(xlabel='Počet příznaků', ylabel='Doba trvání (s)',
       title='Vytvoření predikce na trénovacích datech')
#ax.set(xlim=(0, 10), ylim=(-2, 2),
ax.legend()
fig.savefig('predikce_synt3.pdf')

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(numbers, elapsed_time_train_sklearn_array, color='green', label='scikit-learn')
ax.plot(numbers, elapsed_time_train_offline_array, color='orange', label='offline')
ax.plot(numbers, elapsed_time_train_online_array, color='pink', label='online')
ax.set(xlabel='Počet příznaků', ylabel='Doba trvání (s)',
       title='Učení modelu na trénovacích datech')
ax.legend()
fig.savefig('uceni_synt3.pdf')

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(numbers, np.array(elapsed_time_train_sklearn_array)+np.array(elapsed_time_predict_sklearn_array), color='green', label='scikit-learn')
ax.plot(numbers, np.array(elapsed_time_train_offline_array)+np.array(elapsed_time_predict_offline_array), color='orange', label='offline')
ax.plot(numbers, np.array(elapsed_time_train_online_array)+np.array(elapsed_time_predict_online_array), color='pink', label='online')
ax.set(xlabel='Počet příznaků', ylabel='Doba trvání (s)',
       title='Učení modelu a vytvoření predikce na trénovacích datech')
ax.legend()
fig.savefig('spolu_synt3.pdf')

print('my fit:', np.mean(np.array(elapsed_time_train_offline_array)/np.array(elapsed_time_train_online_array)))
print('sklearn fit:', np.mean(np.array(elapsed_time_train_sklearn_array)/np.array(elapsed_time_train_online_array)))
print('my predict:', np.mean(np.array(elapsed_time_predict_offline_array)/np.array(elapsed_time_predict_online_array)))
print('sklearn predict:', np.mean(np.array(elapsed_time_predict_sklearn_array)/np.array(elapsed_time_predict_online_array)))
print('my overal:', np.mean((np.array(elapsed_time_train_offline_array)+np.array(elapsed_time_predict_offline_array))/(np.array(elapsed_time_train_online_array)+np.array(elapsed_time_predict_online_array))))
print('sklearn overal:', np.mean((np.array(elapsed_time_train_sklearn_array)+np.array(elapsed_time_predict_sklearn_array))/(np.array(elapsed_time_train_online_array)+np.array(elapsed_time_predict_online_array))))'''