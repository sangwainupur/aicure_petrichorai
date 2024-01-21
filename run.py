from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import pickle

regressor=pickle.load(open("check_new_fin.pkl",'rb'))

# print(type(regressor))
# model_loaded.predict(X_test)

#TEST DATA

test_file_path = "merged_data.csv"
test_data = pd.read_csv(test_file_path)

test_data = test_data.drop('datasetId', axis=1)
X_test = test_data.drop('HR', axis=1)
X_test=X_test.drop('uuid', axis=1)  # Features (all columns except 'HR' and 'uuid')
y = test_data['HR']

categorical_columns = ['condition']
condition_index = X_test.columns.get_loc('condition')
print(condition_index)
preprocessor = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(),[condition_index])
    ],
    remainder='passthrough'  # Keep the non-categorical columns as they are
)
X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=preprocessor.get_feature_names_out())

preds=regressor.predict(X_test)

print(preds)

