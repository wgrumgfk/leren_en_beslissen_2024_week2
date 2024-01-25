from sklearn import tree
from matplotlib import pyplot as plt
import pandas as pd
import numpy
from statistics import mean
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

def watt_to_pace(watt):
    # if isinstance(watt, int):
    return float(2000 * (float(2.8/float(watt))**(1/3)))

# PREPARING ##################################################################################################################

# Make sure all rows are complete
df = pd.read_csv('okeanos_processed.csv')
df.dropna(how = 'any', subset = ['days_until_2k','man','zwaar','AT','I','ID','ED','rust_sec','afstand','ervaring','500_split_watt','aantal_intervallen', 'calculated_distance', 'two_k_watt'], inplace=True)

# Define features (X)
# feature_cols = ['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_sec','interval_afstand', 'ervaring', 'rust_sec']
feature_cols = ['days_until_2k','man','zwaar','AT','I','ID','ED','rust_sec','afstand','ervaring','500_split_watt','aantal_intervallen', 'calculated_distance']
X = df[feature_cols]


y = df[['two_k_watt']]


for iter in range(1, 1001):
    accuracies = []

    rand_seed = random.randint(0, 100000)
    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=rand_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rand_seed)

    regr = DecisionTreeRegressor(max_depth=7)
    regr = regr.fit(X_train, y_train)

    # Predict
    y_val_pred = regr.predict(X_val)
    y_train_pred = regr.predict(X_train)

    # Option 1: convert back to seconds
    y_train_pred = numpy.array([watt_to_pace(x) for x in y_train_pred])
    y_train = numpy.vectorize(watt_to_pace)(y_train) 

    y_val_pred = numpy.array([watt_to_pace(x) for x in y_val_pred])
    y_val = numpy.vectorize(watt_to_pace)(y_val) 

    threshold = 10

    good = 0
    for i in range(len(y_val)):
        if abs(y_val[i] - y_val_pred[i]) < threshold:
            good += 1

    accuracies.append(good / len(y_val))

print('accuracy with threshold of', threshold, ':', mean(accuracies))