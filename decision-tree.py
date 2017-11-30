# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from utils import commons

# %%
def classifier_config():
    param_grid = {"clf__criterion": ["gini", "entropy"],
              "clf__min_samples_split": [50, 100, 150],
              "clf__max_depth": [None, 20, 32, 45],
              "clf__min_samples_leaf": [100, 300, 600],
              "clf__max_leaf_nodes": [None, 10],
              "clf__max_features": [50],
              }

    return {
         'classifier': DecisionTreeClassifier(),
         'parameters': param_grid
      }

# %%
def viz_tree(clf):
    num_of_columns = clf.n_features_
    var_vec = [i for i in range(num_of_columns)]
    feature_names = ["_".join(["X", str(i)]) for i in
                 range(1,num_of_columns + 1)]
    #feature_names.insert(0, "digit")
    commons.visualize_tree(clf, feature_names, fn="model_tree_viz")


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
    print("\n")
    viz_tree(model)
    print("\n")
    commons.score(model, x_test, y_test)


# %%
print("====  Start =====")
reload(commons)
run()
print("====  End =====")
