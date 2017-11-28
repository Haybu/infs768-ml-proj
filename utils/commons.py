# Standard scientific Python imports
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from time import time
import numpy as np
from operator import itemgetter

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

# ------------ commons --------

# %%
def perform_grid_search(clf, params, xtrain, ytrain):
    pipe = pipeline.Pipeline([
                            #     ('scaler', preprocessing.StandardScaler()),
                                 ('selector', feature_selector()),
                                 ('clf', clf)
                              ])
    folds = 5
    verbose = 2
    scoring = 'accuracy'

    gs = GridSearchCV(estimator=pipe, param_grid=params
                     , scoring=scoring, cv=folds, verbose=verbose)

    start = time()
    gs.fit(xtrain, ytrain)
    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(gs.grid_scores_)))
    top_params = report(gs.grid_scores_, 3)
    print("\n\n-- Best Parameters:")
    for k, v in top_params.items():
        print("parameter: {:<20s} setting: {}".format(k, v))
    print("\n")
    # manually extract the best models from the grid search to re-build the pipeline
    best_clf = gs.best_estimator_.named_steps['clf']
    print("Best Estimator: {}\n".format(best_clf))
    return best_clf

# %%
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


def plot_histogram(y):
    import matplotlib.pyplot as plt
    import numpy as np; np.random.seed(1)
    a = np.random.rayleigh(scale=3,size=100)
    bins = np.arange(10)
    frq, edges = np.histogram(a, bins)
    fig, ax = plt.subplots()
    ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
    plt.show()

def frequency(y):
    from scipy.stats import itemfreq
    return itemfreq(y)

def preprocess(features, labels):
    from sklearn import preprocessing as pp
    from sklearn.utils import shuffle
    transformed_features = pp.MinMaxScaler().fit_transform(features)
    x, y = shuffle(transformed_features, labels, random_state=0)
    return x, y

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

def digits_data_slim():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    xtrain, xtest, ytrain, ytest = train_test_split(x, y
                               , test_size=0.33, random_state=42)
    return {'train': {'x': xtrain,
                      'y': ytrain
                    },
            'test': {'x': xtest,
                     'y': ytest
                    }
            }

def fold_data(x, y, train_index, test_index):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return {'train': {'x': x_train,
                      'y': y_train
                    },
            'test': {'x': x_test,
                     'y': y_test
                    }
            }

def feature_selector():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    rating_model = ExtraTreesClassifier()
    return SelectFromModel(rating_model, prefit=False)

def selected_features_in_fold(x, y, train_index, test_index):
    x_fold_train, x_fold_test = x[train_index], x[test_index]
    y_fold_train, y_fold_test = y[train_index], y[test_index]
    selector = feature_selector()
    x_fold_train_selected = selector.fit_transform(x_fold_train, y_fold_train)
    x_fold_train_selected_indices = selector.get_support(indices=True)  # a mask
    x_fold_test_selected = x_fold_test[:,x_fold_train_selected_indices]
    return {'train': {'x': x_fold_train_selected,
                      'y': y_fold_train
                    },
            'test': {'x': x_fold_test_selected,
                     'y': y_fold_test
                    }
            }

'''
    calculate the HOG features for each image in the database and save them
    in another numpy array named hog_feature
'''
def hog(features):
    from skimage.feature import hog
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9,
                 pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualise=False)
        list_hog_fd.append(fd)
    return np.array(list_hog_fd, 'float64')

def plot(xdata, ydata, xlabel, ylabel):
    plt.plot(xdata, ydata)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def hog_slim(features):
    from skimage.feature import hog
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((8, 8)), orientations=9,
                 pixels_per_cell=(4, 4), cells_per_block=(1, 1),
                 visualise=False)
        list_hog_fd.append(fd)
    return np.array(list_hog_fd, 'float64')

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

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters

#------ end commons -----------


def show_some_digits(images, targets, sample_size=24, title_text='Digit {}'):
    '''
    Visualize random digits in a grid plot
    images - array of flatten gidigs [:,784]
    targets - final labels
    '''
    nsamples = sample_size
    rand_idx = np.random.choice(images.shape[0], nsamples)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))

    img = plt.figure(1, figsize=(15, 12), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples / 6.0), 6, index + 1)
        plt.axis('off')
        # each image is flat, we have to reshape to 2D array 28x28-784
        plt.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(title_text.format(label))


def plot_confusion_matrix(expected, predicted, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots confusion matrix,

    cm - confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(expected, predicted)
    plt.figure(1, figsize=(15, 12), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots confusion matrix,

    cm - confusion matrix
    """
    plt.figure(1, figsize=(15, 12), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_param_space_scores(scores, C_range, gamma_range):
    """
    Draw heatmap of the validation accuracy as a function of gamma and C


    Parameters
    ----------
    scores - 2D numpy array with accuracies

    """
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.


    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.jet,
               norm=MidpointNormalize(vmin=0.5, midpoint=0.9))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
