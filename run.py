import pandas as pd
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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

def load_model():
    # Load the pre-trained model
    model = pickle.load(open("check_new_fin.pkl",'rb'))
    return model

def make_predictions(model,test_data):
    test_data = test_data.drop('datasetId', axis=1)
    X_test=test_data.drop('uuid', axis=1)  


    categorical_columns = ['condition']
    condition_index = X_test.columns.get_loc('condition')
    # print(condition_index)
    preprocessor = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(),[condition_index])
    ],
    remainder='passthrough'  # Keep the non-categorical columns as they are
    )
    X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=preprocessor.get_feature_names_out())

    preds=model.predict(X_test)
    # print(preds)
    return preds

def main():
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("input_file", help="Path to the input file (test_data.csv)")
    args = parser.parse_args()

    input_data = pd.read_csv(args.input_file)

    model=load_model()
    predictions=make_predictions(model,input_data)
    ids = input_data['uuid']

    output = pd.DataFrame({'uuid': ids,
                       'HR': predictions.squeeze()})
    
    output.to_csv('results.csv', index=False)




if __name__ == "__main__":
    main()