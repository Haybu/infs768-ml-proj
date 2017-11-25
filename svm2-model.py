# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import time
import datetime as dt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from io import BytesIO
from operator import itemgetter
from utils import commons

# %%

SLIM = False

def test_fit(features, labels):
    param_C = 5
    param_gamma = 0.05
    classifier = svm.SVC(C=param_C,gamma=param_gamma)
    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    classifier.fit(features, labels)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

# %%
def fit(features, labels, parameters):
    svc = SVC()
    grid = GridSearchCV(estimator=svc,param_grid=parameters,n_jobs=1, verbose=2)
    start_time = dt.datetime.now()
    print('Start param searching at {}'.format(str(start_time)))
    grid.fit(features, labels)
    elapsed_time= dt.datetime.now() - start_time
    print('Elapsed time, param searching {}'.format(str(elapsed_time)))
    sorted(grid.cv_results_.keys())
    return grid

def restore_grid():
    filename = "/Users/hxm3459/temp/models/svm.sav"
    return joblib.load(filename)

# %%
def grid_mean_test_score(grid, c_range, gamma_range):
    scores = grid.cv_results_['mean_test_score'].reshape(len(c_range),
                                                     len(gamma_range))
    commons.plot_param_space_scores(scores, c_range, gamma_range)
    print(scores)

# %%
def evaluate(features, expected, grid):
    from sklearn import metrics
    classifier = grid.best_estimator_
    params = grid.best_params_
    print("Classifier best parameters {}".format(params))
    predicted = classifier.predict(features)
    #show_some_digits(features,predicted,title_text="Predicted {}")
    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))
    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)
    commons.plot_confusion_matrix(cm)
    print("Accuracy = {}".format(metrics.accuracy_score(expected, predicted)))


# %%
def main():
    reload(commons)
    data = commons.digits_data_slim() if (SLIM == True) else commons.digits_data()
    xtrain_data = data['train']['x']
    xtrain = commons.hog_slim(xtrain_data) if (SLIM == True) else commons.hog(xtrain_data)
    ytrain = data['train']['y']
    xtest_data = data['test']['x']
    xtest = commons.hog_slim(xtest_data) if (SLIM == True) else commons.hog(xtest_data)
    ytest = data['test']['y']

    # models parameters
    gamma_range = np.outer(np.logspace(-3, 0, 4),np.array([1,5]))
    gamma_range = gamma_range.flatten()
    C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5]))
    C_range = C_range.flatten()
    parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}

    grid = fit(xtrain, ytrain,parameters)

    filename = "/Users/hxm3459/temp/models/svm.sav"
    joblib.dump(grid, filename)

    grid_mean_test_score(grid, C_range, gamma_range)
    evaluate(xtest, ytest, grid)

# %%
def main_saved():
    filename = "/Users/hxm3459/temp/models/svm.sav"
    loaded_model = joblib.load(filename)
    grid_mean_test_score(grid, C_range, gamma_range)
    evaluate(xtest, ytest, grid)

# %%
def main_test():
    data = commons.get_data()
    xtrain_data = data['train']['x']
    xtrain = commons.hog(xtrain_data)
    ytrain = data['train']['y']
    xtest_data = data['test']['x']
    xtest = commons.hog(xtest_data)
    ytest = data['test']['y']

    test_fit(xtrain, ytrain)


# %%
print("Start SVM modeling")
main()
print("End SVM modeling")
