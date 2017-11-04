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
