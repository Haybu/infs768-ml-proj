# %%
from sklearn.neural_network import MLPClassifier
from utils import commons

# %%
def classifier_config():
    param_grid = {
        'clf__learning_rate': ["constant", "invscaling", "adaptive"],
        'clf__hidden_layer_sizes': [(50,50,10),(128,128,10),(150,150,10)],
        'clf__alpha': [0.01, 0.001, 0.0001],
        'clf__activation': ["logistic", "relu", "tanh"]
        }

    return {
         'classifier': MLPClassifier(),
         'parameters': param_grid
      }

# %%
def run():
    data = commons.digits_data_slim(); # commons.digits_data()
    x_train = data['train']['x']
    y_train = data['train']['y']
    x_test = data['test']['x']
    y_test = data['test']['y']

    config = classifier_config()
    grid = commons.gridCV(config['classifier'], config['parameters'], x_train, y_train)
    commons.score(grid, x_test, y_test)

# %%
print("====  Start =====")
reload(commons)
run()
print("====  End =====")
