# %%
from sklearn.neural_network import MLPClassifier
from utils import commons

# %%
def classifier_config():
    param_grid = [{
                  'clf__C': [0.00001, 0.0001, 0.001, 0.01 , 0.1],
                  'clf__solver': [ 'lbfgs', 'newton-cg', 'sag'],
                  'clf__multi_class': ['multinomial'],
                  'clf__penalty': ['l2']
                  }]

    return {
         'classifier': linear_model.LogisticRegression(),
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
