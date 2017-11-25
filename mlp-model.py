# %%
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from utils import commons

# %%

def main():
    data = commons.digits_data()

    x_train_data = data['train']['x']
    x_train = x_train_data / 255.
    y_train = data['train']['y']

    x_test_data = data['test']['x']
    x_test = x_test_data / 255.
    y_test = data['test']['y']
    
    parameters={
        'learning_rate': ["constant", "invscaling", "adaptive"],
        'hidden_layer_sizes': [(50,50,10),(128,128,10),(150,150,10)],
        'alpha': [0.01, 0.001, 0.0001],
        'activation': ["logistic", "relu", "tanh"]
        }

    classifier = MLPClassifier()
    clf = GridSearchCV(estimator=classifier,param_grid=parameters,n_jobs=-1,verbose=2,cv=10)
    clf.fit(x_train, y_train)
    print("Best Parameters: {}".format(clf.best_params_))
    print("Best Estimator: {}".format(clf.best_estimator_))
    print("Training set score: %f" % clf.score(x_train, y_train))
    print("Test set score: %f" % clf.score(x_test, y_test))
    y_predict = clf.predict(x_test)
    commons.print_confusion_matrix(y_test, y_predict)
    commons.print_classification_report(y_test, y_predict)

# %%
reload(commons)
main()
