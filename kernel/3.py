import numpy as np,\
pandas as pd,\
seaborn as sns,\
matplotlib.pyplot as plt,\
sklearn.preprocessing as Scaler

import sklearn
from sklearn import linear_model
from sklearn import preprocessing

from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline

pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))

import sqlalchemy
from sqlalchemy import create_engine

import pickle

engine = create_engine("sqlite:////home/yash/kaggle/house.db", echo=False)

training_data = pd.read_sql_table('traindata', con = engine).drop(columns = ['index'])
Y = pd.read_sql_table('Y', con = engine).drop(columns = ['index'])

# Final Lasso model
#
############################
training_model = sklearn.linear_model.LassoCV(copy_X = True, fit_intercept = True,
                                            alphas = [x * .00001 for x in range(1,200, 2)], 
                                                             tol = 0.0000000001, max_iter = 1000000000)
training_model.fit(training_data, Y['SalePrice'])
pickle.dump(training_model, open('model.p', 'wb'))

print("Training DONE!!!")
