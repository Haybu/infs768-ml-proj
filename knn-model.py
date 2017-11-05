'''
This program implements a KNN model to predict handwritten digits.
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
'''
Examine different values of K, using cross validation with multiple folds
'''
def cross_validate(features, labels, kmax, cv):
    print("Searching a KNN model with kmax {} and cv {}".format(kmax, cv))
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
            x_fold_train = fold['train']['x']
            y_fold_train = fold['train']['y']
            x_fold_test = fold['test']['x']
            y_fold_test = fold['test']['y']
            knn.fit(x_fold_train, y_fold_train)
            score = knn.score(x_fold_test, y_fold_test)
            folds_scores.append(score)
            print("------ Complete scoring knn model with K = {} in fold number {} ---------".format(k, fold_number))

        print("Train completed for k = {}".format(k))
        avg = sum(folds_scores) / len(folds_scores)
        model_scores.append(tuple([k, avg]))
        print("Average model score for k {} is {}".format(k, avg))

    print("Searching completed ...{}".format(model_scores))
    return model_scores

def best_k(tpl):
    model_k = max(tpl,key=itemgetter(1))[0]
    print("The best k is {}".format(model_k))
    return model_k

def plot_k_accuracy(tpl):
    k_values, avg_values = zip(*tpl)  # the * sign unpack the argument to it elements to pass over to the method
    commons.plot(k_values, avg_values, 'Number of Neighbors K', 'Accuracy Score')


def main():
    K_MAX = 15
    NUMBER_OF_FOLDS = 10

    print("====  Start =====")
    data = commons.digits_data()

    x_train_data = data['train']['x']
    x_train = commons.hog(x_train_data)
    y_train = data['train']['y']

    x_test_data = data['test']['x']
    x_test = commons.hog(x_test_data)
    y_test = data['test']['y']

    #show some sample images
    #commons.show_some_digits(x_train_data, y_train)

    print("start tuning k")
    k_scores = cross_validate(features=x_train, labels=y_train, kmax=K_MAX, cv=NUMBER_OF_FOLDS)
    k = best_k(k_scores)
    #k_scores = [(2, 0.86236911556906082), (3, 0.87125998528289839), (4, 0.87510151565077765), (5, 0.87660986230921922), (6, 0.87820064155578592), (7, 0.87910605329375335), (8, 0.88053135811604444), (9, 0.88140865992705586)]
    #k = 9

    print("start training the knn model with training data")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    print("start scoring the model with the testing data")
    y_predict = knn.predict(x_test)

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
