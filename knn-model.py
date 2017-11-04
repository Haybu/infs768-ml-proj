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

# %%
def load_digits_data():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    y = mnist.target
    return X, y

def digits_data():
    X, y = load_digits_data()
    X_train = X[0:60000,:]
    y_train = y[0:60000]
    X_test = X[60000:,:]
    y_test = y[60000:]
    return {
        'x_train': X_train,
        'y_train': y_train,
        'x_test': X_test,
        'y_test': y_test
    }

def k_range(max):
    return range(2, max)

def feature_selecter():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    rating_model = ExtraTreesClassifier()
    return SelectFromModel(rating_model, prefit=False)

def selected_features_in_fold(x, y, train_index, test_index):
    x_fold_train, x_fold_test = x[train_index], x[test_index]
    y_fold_train, y_fold_test = y[train_index], y[test_index]
    selecter = feature_selecter()
    x_fold_train_selected = selecter.fit_transform(x_fold_train, y_fold_train)
    x_fold_train_selected_indices = selecter.get_support(indices=True)  # a mask
    x_fold_test_selected = x_fold_test[:,x_fold_train_selected_indices]
    return {
        'x_fold_train_selected': x_fold_train_selected,
        'x_fold_test_selected': x_fold_test_selected,
        'y_fold_train': y_fold_train,
        'y_fold_test': y_fold_test
    }

def examine_k(features, labels, kmax, cv):
    print("Examining a KNN model with kmax {} and cv {}".format(kmax, cv))
    from sklearn.model_selection import StratifiedKFold
    folds_scores = []
    model_scores = []
    skf = StratifiedKFold(n_splits=cv)

    for k in k_range(kmax):
        knn = KNeighborsClassifier(n_neighbors=k)
        fold_number = 0
        for train_index, test_index in skf.split(features, labels):
            fold_number += 1
            print("------ Starting scoring knn model with K = {} in fold number {} ---------".format(k, fold_number))
            fold = selected_features_in_fold(features, labels, train_index, test_index)
            x_fold_train_selected = fold['x_fold_train_selected']
            y_fold_train = fold['y_fold_train']
            x_fold_test_selected = fold['x_fold_test_selected']
            y_fold_test = fold['y_fold_test']
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
    plot(k_values, avg_values, 'Number of Neighbors K', 'Accuracy Score')


def plot(xdata, ydata, xlabel, ylabel):
    plt.plot(xdata, ydata)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def accuracy_score(expected, predicted):
    from sklearn.metrics import accuracy_score
    return accuracy_score(expected, predicted)

def print_confusion_matrix(expected, predicted):
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(expected, predicted))

def print_classification_report(expected, predited):
    from sklearn.metrics import classification_report
    print(classification_report(expected, predited))

# %%
def main():
    K_MAX = 7
    NUMBER_OF_FOLDS = 5

    print("====  Start =====")
    data = digits_data()
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    print("start tuning k")
    #k_scores = examine_k(features=x_train, labels=y_train, kmax=K_MAX, cv=NUMBER_OF_FOLDS)
    #k = best_k(k_scores)
    k_scores = [(2, 0.96273320002053331), (3, 0.96578327108267836), (4, 0.96637210862977629), (5, 0.96683318610161617), (6, 0.96689318617835784)]
    k = 6

    print("start training the knn model with training data")
    knn = KNeighborsClassifier(n_neighbors=k)

    selecter = feature_selecter()
    x_train_selected = selecter.fit_transform(x_train, y_train)  # features selection
    x_train_selected_indices = selecter.get_support(indices=True)
    knn.fit(x_train_selected, y_train)

    x_test_selected = x_test[:,x_train_selected_indices]

    print("start scoring the model with the testing data")
    y_predict = knn.predict(x_test_selected)

    model_accuracy_score = accuracy_score(y_test, y_predict)
    print("Final model score is {}".format(model_accuracy_score))

    print_confusion_matrix(y_test, y_predict)
    print_classification_report(y_test, y_predict)
    plot_k_accuracy(k_scores)


    #print("Testing...")
    #test1(knn)
    #test2(x_train, y_train, 200, knn)

    print("====  Done =====")

# %%
def test1(model):
    # predict a random digit
    nums = np.random.choice([0, 1], size=784)
    nums = nums.reshape(1, 784)  # convert to a 2d array
    #print(nums)
    digit_predict = model.predict(nums)
    print("prediction {}\n".format(digit_predict))

# %%
def test2(features, labels, image_index, model):
    x = features[image_index,:]
    y = labels[image_index]
    prediction = model.predict(x.reshape(1,784))
    print("Expected y is {} , predicted as {}".format(y, prediction))

# %%
#if __name__ == '__main__':
main()
