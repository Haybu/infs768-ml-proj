# %%
from sklearn.linear_model import LogisticRegression
from utils import commons

# %%
def classifier_config():
    param_grid = [{
                  'selector__k_features': [392,471, 706],  #50%, 60%, 90%
                  'selector__estimator__C': [0.00001, 0.0001, 0.001, 0.01 , 0.1],
                  'selector__estimator__solver': [ 'lbfgs', 'newton-cg', 'sag'],
                  'selector__estimator__multi_class': ['multinomial'],
                  'selector__estimator__penalty': ['l2']
                  }]

    param_grid_slim = [{
                   'selector__k_features': ["best"],  #50%, 60%, 90%
                   'selector__estimator__C': [0.0001],
                   'selector__estimator__solver': [ 'lbfgs'],
                   'selector__estimator__multi_class': ['multinomial'],
                   'selector__estimator__penalty': ['l2']
                      }]

    return {
         'classifier': LogisticRegression(),
         'parameters': param_grid_slim
      }

# %%
def run():
    data = commons.digits_data() # commons.digits_data_slim()
    x_train = data['train']['x'] / 255.
    y_train = data['train']['y']
    x_test = data['test']['x'] / 255.
    y_test = data['test']['y']

    config = classifier_config()

    selector = commons.sfs_logit_reg(config['classifier'])

    model = commons.perform_grid_search(config['classifier']
                        , config['parameters']
                        , selector
                        , x_train
                        , y_train
                         )

    commons.score(model, x_test, y_test)

# %%
print("====  Start =====")
reload(commons)
run()
print("====  End =====")
