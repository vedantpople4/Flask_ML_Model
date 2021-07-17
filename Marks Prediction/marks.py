import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def marks_prediction(marks):
    x = pd.read_csv("test.csv")
    y = pd.read_csv("train.csv")
    x = x.values
    y = y.values
    model = LinearRegression()
    model.fit(x,y)
    x_test = np.array(marks, dtype='float64')
    x_test = x_test.reshape((1,-1))
    return model.predict(x_test)[0]
