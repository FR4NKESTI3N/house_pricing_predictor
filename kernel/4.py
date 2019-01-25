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

model = pickle.load(open('model.p', 'rb'))

test_data = pd.read_sql_table('testdata', con = engine).drop(columns = ['index'])

prediction = pd.DataFrame()
prediction['SalePrice'] = np.expm1(model.predict(test_data))
prediction.to_sql('prediction', con = engine, if_exists = 'replace')

print("prediction done and loaded to database!!")
