# %%
from sklearn.linear_model import LogisticRegression
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
         'classifier': LogisticRegression(),
         'parameters': param_grid
      }

# %%
def run():
    data = commons.digits_data() # commons.digits_data_slim()
    x_train = data['train']['x'] / 255.
    y_train = data['train']['y']
    x_test = data['test']['x'] / 255.
    y_test = data['test']['y']

    config = classifier_config()
    selector = commons.feature_selector_chi2(90)
    model = commons.perform_grid_search(config['classifier']
                        , config['parameters']
                        , selector
                        , x_train
                        , y_train)
    commons.score(model, x_test, y_test)

# %%
print("====  Start =====")
reload(commons)
run()
print("====  End =====")
