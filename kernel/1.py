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
# import mysql.connector as sql
engine = create_engine("sqlite:////home/yash/kaggle/house.db", echo=False)

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train_data.to_sql('train', con = engine, if_exists = 'replace')
test_data.to_sql('test', con = engine, if_exists = 'replace')

print("Files loaded to SQL database")
