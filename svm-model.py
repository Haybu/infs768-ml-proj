# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from matplotlib.pyplot import show, imshow, cm
from sklearn.svm import SVC

%matplotlib inline


# %%
def analyze(clf, data):
    """
    Analyze how well a classifier performs on data.

    Parameters
    ----------
    clf : classifier object
    data : dict
    """
    # Get confusion matrix
    predicted = clf.predict(data['test']['X'])
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(data['test']['y'],
                                   predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(data['test']['y'],
                                                     predicted))

    # Print example
    try_id = 1
    out = clf.predict(data['test']['X'][try_id].reshape(1, -1))  # clf.predict_proba
    print("out: %s" % out)
    size = int(len(data['test']['X'][try_id])**(0.5))
    view_image(data['test']['X'][try_id].reshape((size, size)),
               data['test']['y'][try_id])

# %%
def view_image(image, label=""):
    """
    View a single image.

    Parameters
    ----------
    image : numpy array
        Make sure this is of the shape you want.
    label : str
    """
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()

# %%
def get_data():
    """
    Get data ready to learn with.

    Returns
    -------
    dict
    """
    simple = False
    if simple:  # Load the simple, but similar digits dataset
        digits = load_digits()
        x = [np.array(el).flatten() for el in digits.images]
        y = digits.target


        # Scale data to [-1, 1] - This is of mayor importance!!!
        # In this case, I know the range and thus I can (and should) scale
        # manually. However, this might not always be the case.
        # Then try sklearn.preprocessing.MinMaxScaler or
        # sklearn.preprocessing.StandardScaler
        #x = x/255.0*2 - 1
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        x, y = shuffle(x, y, random_state=0)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)
        data = {'train': {'X': x_train,
                          'y': y_train},
                'test': {'X': x_test,
                         'y': y_test}}
    else:  # Load the project dataset
        mnist = fetch_mldata('MNIST original')
        x = mnist.data
        y = mnist.target

        print("x shape {}".format(x.shape))
        print("y shape {}".format(y.shape))

        # Scale data to [-1, 1] - This is of major importance!!!
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        x, y = shuffle(x, y, random_state=0)

        from sklearn.cross_validation import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)
        data = {'train': {'X': x_train,
                          'y': y_train},
                'test': {'X': x_test,
                         'y': y_test}}
    return data

# %%
"""Orchestrate the retrival of data, training and testing."""
data = get_data()

# Get classifier
clf = SVC(probability=False,  # cache_size=200,
        kernel="rbf", C=2.8, gamma=.0073)

print("Start fitting. This may take a while")

# take all of it - make that number lower for experiments
examples = len(data['train']['X']) / 70
clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])

analyze(clf, data)

# %%
