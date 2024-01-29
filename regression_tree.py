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
from sklearn.model_selection import cross_val_score, KFold

def watt_to_pace(watt):
    try:
        return float(2000 * (float(2.8/float(watt))**(1/3)))
    except (ValueError, ZeroDivisionError):
        return numpy.nan  # Handle non-numeric values or division by zero gracefully

# PREPARING ##################################################################################################################

# Make sure all rows are complete
df = pd.read_csv('okeanos_processed.csv')
df.dropna(how = 'any', subset = ['days_until_2k','man','zwaar','AT','I','ID','ED','rust_sec','afstand','ervaring','500_split_watt','aantal_intervallen', 'calculated_distance', 'two_k_watt'], inplace=True)

# Define features (X)
# feature_cols = ['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_sec','interval_afstand', 'ervaring', 'rust_sec']
feature_cols = ['days_until_2k','man','zwaar','AT','I','ID','ED','rust_sec','afstand','ervaring','500_split_watt','aantal_intervallen', 'calculated_distance']
X = df[feature_cols]
y = df[['two_k_watt']]



######################################################################################################################################## Best value cross-validation
# Initialize variables to track the best model
best_score = float('inf')  # initialize with a large value

######################################################################################################################################


for iter in range(1, 1001):
    accuracies = []

    rand_seed = random.randint(0, 100000)
    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=rand_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rand_seed)

    regr = DecisionTreeRegressor(max_depth=7)
    regr = regr.fit(X_train, y_train)

#################################################################################################################################### Perform cross-validation
    num_folds = 5  # You can adjust the number of folds as needed
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=rand_seed)

    # Convert the target variable to seconds for cross-validation
    if 'two_k_watt' in y.columns:
        y_sec = numpy.array([watt_to_pace(x) for x in y['two_k_watt'].iloc[1:]])
    else:
        y_sec = y.iloc[1:]

    # Ensure X and y_sec have the same length
    X = X.iloc[:len(y_sec)]

    # Calculate cross-validated scores on the converted target variable
    cv_scores_sec_tree = cross_val_score(regr, X, y_sec, cv=kf, scoring='neg_mean_squared_error')
    cv_scores_sec_tree = -cv_scores_sec_tree  # Convert back to positive values

    # Compute the mean of the cross-validated scores
    mean_cv_score_sec_tree = numpy.mean(cv_scores_sec_tree)

        # Check if the current model is the best
    if mean_cv_score_sec_tree < best_score:
        best_score = mean_cv_score_sec_tree
    
################################################################################################################################################
    # Predict
    y_val_pred = regr.predict(X_val)
    y_train_pred = regr.predict(X_train)

    # Option 1: convert back to seconds
    y_train_pred = numpy.array([watt_to_pace(x) for x in y_train_pred])
    y_train = numpy.vectorize(watt_to_pace)(y_train) 

    y_val_pred = numpy.array([watt_to_pace(x) for x in y_val_pred])
    y_val = numpy.vectorize(watt_to_pace)(y_val) 


    threshold = 5
    good = 0
    for i in range(len(y_val)):
        if abs(y_val[i] - y_val_pred[i]) < threshold:
            good += 1

    accuracies.append(good / len(y_val))




print('accuracy with threshold of', threshold, ':', mean(accuracies))

# Print the best cross-validated MSE
print("Best Cross-validated MSE for Decision Tree: ", best_score)