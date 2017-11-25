'''
This program implements training and testing a KNN model
'''

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from io import BytesIO
from operator import itemgetter
from utils import commons

# %%
def cross_validate(features, labels, kmax, cv):
    print("Examining a KNN model with kmax {} and cv {}".format(kmax, cv))
    from sklearn.model_selection import StratifiedKFold
    folds_scores = []
    model_scores = []
    skf = StratifiedKFold(n_splits=cv)

    for k in range(2, kmax):
        knn = KNeighborsClassifier(n_neighbors=k)
        fold_number = 0
        for train_index, test_index in skf.split(features, labels):
            fold_number += 1
            print("------ Starting scoring knn model with K = {} in fold number {} ---------".format(k, fold_number))
            fold = commons.selected_features_in_fold(features, labels
                         , train_index, test_index)
            x_fold_train_selected = fold['train']['x']
            y_fold_train = fold['train']['y']
            x_fold_test_selected = fold['test']['x']
            y_fold_test = fold['test']['y']
            knn.fit(x_fold_train_selected, y_fold_train)
            score = knn.score(x_fold_test_selected, y_fold_test)
            folds_scores.append(score)
            print("------ Complete scoring knn model with K = {} in fold number {} ---------".format(k, fold_number))

        print("Train completed for k = {}".format(k))
        avg = sum(folds_scores) / len(folds_scores)
        model_scores.append(tuple([k, avg]))
        print("Average model score for k {} is {}".format(k, avg))

    print("Examining completed ...{}".format(model_scores))
    return model_scores

def best_k(tpl):
    model_k = max(tpl,key=itemgetter(1))[0]
    print("The best k is {}".format(model_k))
    return model_k

def plot_k_accuracy(tpl):
    k_values, avg_values = zip(*tpl)  # the * sign unpack the argument to it elements to pass over to the method
    commons.plot(k_values, avg_values, 'Number of Neighbors K', 'Accuracy Score')


# %%
def main():
    K_MAX = 10
    NUMBER_OF_FOLDS = 5

    print("====  Start =====")
    data = commons.digits_data()
    x_train = data['train']['x']
    y_train = data['train']['y']
    x_test = data['test']['x']
    y_test = data['test']['y']

    print("start tuning k")
    k_scores = cross_validate(features=x_train, labels=y_train, kmax=K_MAX, cv=NUMBER_OF_FOLDS)
    k = best_k(k_scores)
    #k_scores = [(2, 0.96273320002053331), (3, 0.96578327108267836), (4, 0.96637210862977629), (5, 0.96683318610161617), (6, 0.96689318617835784)]
    #k = 6

    print("start training the knn model with training data")
    knn = KNeighborsClassifier(n_neighbors=k)

    selector = commons.feature_selector()
    x_train_selected = selector.fit_transform(x_train, y_train)  # features selection
    x_train_selected_indices = selector.get_support(indices=True)
    knn.fit(x_train_selected, y_train)

    x_test_selected = x_test[:,x_train_selected_indices]

    print("start scoring the model with the testing data")
    y_predict = knn.predict(x_test_selected)

    model_accuracy_score = commons.accuracy_score(y_test, y_predict)
    print("Final model score is {}".format(model_accuracy_score))

    commons.print_confusion_matrix(y_test, y_predict)
    commons.print_classification_report(y_test, y_predict)
    plot_k_accuracy(k_scores)

    print("====  Done =====")

# %%
#if __name__ == '__main__':
reload(commons)
main()
