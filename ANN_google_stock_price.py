# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a RNN tutorial from Udemy.
Google stock price prediction using Deep learning

"""

# Data Preprocessing
# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the training set
x_train = pd.read_csv("datas/Google_Stock_train.csv")
x_train.head()
x_test = pd.read_csv("datas/Google_Stock_test.csv")

# Feature Scalling
from sklearn.preprocessing import  MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(x_train)