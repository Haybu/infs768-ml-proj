
# %%
def load_digits_data():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    x = mnist.data
    y = mnist.target
    return x, y

def preprocess(features, labels):
    from sklearn import preprocessing as pp
    from sklearn.utils import shuffle
    transformed_features = pp.MinMaxScaler().fit_transform(features)
    x, y = shuffle(transformed_features, labels, random_state=0)
    return x, y

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
    hog_features = np.array(list_hog_fd, 'float64')
    return hog_features

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

# %%
import numpy as np
data = digits_data()
xtrain = data['train']['x']
ytrain = data['train']['y']
print(xtrain.shape)
print(ytrain.shape)
print(type(xtrain))
print(type(ytrain))
