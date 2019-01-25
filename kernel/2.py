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

pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))

import sqlalchemy
from sqlalchemy import create_engine


engine = create_engine("sqlite:////home/yash/kaggle/house.db", echo=False)

train_data = pd.read_sql('train', con = engine).drop(columns = ['index'])
test_data = pd.read_sql('test', con = engine).drop(columns = ['index'])

features_to_drop = ['GarageArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'Utilities']

features_to_trans = ['MiscVal', 'PoolArea', 'LotArea', '3SsnPorch', 'LowQualFinSF',
                     'KitchenAbvGr', 'BsmtFinSF2', 'ScreenPorch', 'BsmtHalfBath',
                     'EnclosedPorch', 'MasVnrArea', 'OpenPorchSF', 'LotFrontage',
                     'BsmtFinSF1', 'WoodDeckSF', '1stFlrSF',
                     'GrLivArea', 'BsmtUnfSF', '2ndFlrSF', 'HalfBath']

features_categorical = ['LotShape', 'OverallQual', 'OverallCond', 'LandContour', 'LotConfig',
                        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                        'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'MSZoning', 'MSSubClass',
                        'Street', 'Alley', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
                        'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                        'GarageType']

features_numerical = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF',
                      '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                      'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'YrSold', 'YearBuilt',
                      'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'GarageYrBlt',
                      'YearRemodAdd','MoSold']

features_na_null = ['GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MSSubClass', 'Alley',
                   'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                   'GarageType']

features_na_zero = ['GarageYrBlt', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath',
                   'MasVnrArea']


# Randomize train_data before splitting
#
########################################
train_data = train_data.sample(frac = 1).reset_index(drop = True)
Y = train_data[['Id', 'SalePrice']]
all_data = pd.concat((train_data.drop(columns = ['SalePrice']), test_data))

# Drop features and fill NA
#
#########################################

all_data = all_data.drop(columns = features_to_drop)

all_data['MSZoning'] = all_data['MSZoning'].fillna('RL')
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr')
all_data['SaleType'] = all_data['SaleType'].fillna('WD')
all_data['KitchenQual'] = all_data['KitchenQual'].fillna('TA')
all_data['Exterior1st'] = all_data['Exterior1st'].fillna('VinylSd')
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna('VinylSd')

all_data[features_na_null] = all_data[features_na_null].fillna("None")

all_data['LotFrontage'] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda col: 
                                                                                    col.fillna(col.dropna().median()))

all_data[features_na_zero] = all_data[features_na_zero].fillna(0)

all_data[features_categorical] = all_data[features_categorical].fillna("None")

all_data[features_numerical] = all_data[features_numerical].fillna(0)


# Apply log transform
#
#########################################

Y['SalePrice'] = np.log1p(Y['SalePrice'])
all_data[features_to_trans] = np.log1p(all_data[features_to_trans])

all_data_dum = pd.get_dummies(all_data.drop('Id', axis = 1))

Y.to_sql('Y', con = engine, if_exists = 'replace')
all_data_dum[:len(train_data)].to_sql('traindata', con = engine, if_exists = 'replace')
all_data_dum[len(train_data):].to_sql('testdata', con = engine, if_exists = 'replace')

print("Data processing done!!")
