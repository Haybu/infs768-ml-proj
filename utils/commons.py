import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from operator import itemgetter
import numpy as np

from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import feature_selection
from sklearn.tree import export_graphviz

import os
import subprocess
from time import time


def perform_grid_search(clf, params, selector, xtrain, ytrain, folds=5):
    pipe = pipeline.Pipeline([
                            #     ('scaler', preprocessing.StandardScaler()),
                                 ('selector', selector),
                                 ('clf', clf)
                              ])
    verbose = 2
    scoring = 'accuracy'

    gs = GridSearchCV(estimator=pipe, param_grid=params
                     , scoring=scoring
                     , cv=folds
                     , verbose=verbose
                     , n_jobs=1)

    start = time()
    gs.fit(xtrain, ytrain)
    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(gs.grid_scores_)))
    top_params = report(gs.grid_scores_, 3)
    print("\n\n-- Best first {} Parameters:".format(3))
    for k, v in top_params.items():
        print("parameter: {:<20s} setting: {}".format(k, v))
    print("\n")
    # manually extract the best models from the grid search to re-build the pipeline
    best_clf = gs.best_estimator_.named_steps['clf']
    print("Best Estimator: {}\n".format(best_clf))
    print("\n")
    print(gs.best_estimator_.steps)
    return best_clf

def score(clf, xtest, ytest):
    pl = pipeline.Pipeline([
                        ('selector', feature_selector()),
                        ('classifier', clf)
                 ])

    # passing gs_clf here would run the grid search again inside cross_val_predict
    y_predicted = cross_val_predict(pl, xtest, ytest)
    scores = cross_val_score(pl, xtest, ytest)
    #print(metrics.classification_report(ytest, y_predicted, digits=3))
    print_confusion_matrix(ytest, y_predicted)
    print("\n")
    print_classification_report(ytest, y_predicted)
    print("\n")
    print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std()) )

def load_digits_data():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    x = mnist.data
    y = mnist.target
    return x, y

def digits_data():
    x, y = load_digits_data()
    features =  np.array(x, 'int16')
    labels = np.array(y, 'int')
    return {'train': {'x': features[0:60000,:],
                      'y': labels[0:60000]
                    },
            'test': {'x': features[60000:,:],
                     'y': labels[60000:]
                    }
            }

def feature_selector_chi2(percent_to_keep):
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import chi2
    # percentile to keep (90% for example)
    return SelectPercentile(score_func=chi2, percentile=percent_to_keep)

def accuracy_score(expected, predicted):
    from sklearn.metrics import accuracy_score
    return accuracy_score(expected, predicted)

def print_confusion_matrix(expected, predicted):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(expected, predicted)
    print(cm)


def print_classification_report(expected, predited):
    from sklearn.metrics import classification_report
    print(classification_report(expected, predited))

def visualize_tree(tree, feature_names, fn="dt"):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn Decision Tree.
    feature_names -- list of feature names.
    fn -- [string], root of filename, default `dt`.
    """
    dotfile = fn + ".dot"
    pngfile = fn + ".png"

    with open(dotfile, 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", dotfile, "-o", pngfile]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, "
             "to produce visualization")
