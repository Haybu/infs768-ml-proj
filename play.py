
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

def digits_data():
    features, labels = load_digits_data()
    x, y = preprocess(features, labels)
    x_train = x[0:60000,:]
    y_train = y[0:60000]
    x_test = x[60000:,:]
    y_test = y[60000:]
    return {'train': {'X': x_train,
                      'y': y_train},
            'test': {'X': x_test,
                     'y': y_test}
            }

# %%
import numpy as np
data = digits_data()
xtrain = data['train']['X']
print(xtrain.shape)
print(type(xtrain))
np.amax(xtrain)
