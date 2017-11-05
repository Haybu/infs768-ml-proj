# Standard scientific Python imports
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


# ------------ commons --------

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

def selected_features_in_fold(x, y, train_index, test_index):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return {'train': {'x': x_train,
                      'y': y_train
                    },
            'test': {'x': x_test,
                     'y': y_test
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
