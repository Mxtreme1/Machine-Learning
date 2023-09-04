""" Boston Housing Market """

#Imports
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import pandas
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error


def pre_process(training_data, test_data):
    with open(training_data) as f:
        train = pd.read_csv(f)
    with open(test_data) as f:
        test = pd.read_csv(f)
    train_X = train.iloc[:, 1:-1]
    train_y = train.medv
    test_X = test.iloc[:, 1:]
    return train, test, train_X, train_y, test_X

def preview_data(train):
    correlation_matrix = train.corr().round(2)
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()
    # for i in range(train.shape[1] - 2):
    #     sns.lmplot(data=train, x=train.columns[i + 1], y="medv")  # +1 to skip ID and get second to last column (last=y)
    #     plt.show()

def train_linear_regression_model(train_X, train_y):
    return LinearRegression().fit(train_X, train_y)

def predict(reg, test_X):
    pred = pd.DataFrame(reg.predict(test_X))
    pred.columns = ["medv"]
    return pred

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_path = os.path.abspath(os.path.join("data", "train.csv"))
    test_path = os.path.abspath(os.path.join("data", "test.csv"))
    output_path = os.path.abspath(os.path.join("data", "output.csv"))
    train, test, train_X, train_y, test_X = pre_process(train_path, test_path)
    # preview_data(train)
    reg = train_linear_regression_model(train_X, train_y)
    pred = predict(reg, test_X)
    pred = pred.set_index(test.iloc[:, 0], drop=True)
    pred.to_csv(output_path)

    print()