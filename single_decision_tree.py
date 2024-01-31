from sklearn import tree
from matplotlib import pyplot as plt
import pandas as pd
import numpy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

regressor = True
wattage = True

def watt_to_pace(watt):
    # if isinstance(watt, int):
    return float(2000 * (float(2.8/float(watt))**(1/3)))

# PREPARING ##################################################################################################################

# Make sure all rows are complete
df = pd.read_csv('okeanos_processed.csv')
df.dropna(how = 'any', subset =  ['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_watt', 'ervaring', 'rust_sec', 'aantal_intervallen', 'calculated_distance', 'afstand', 'two_k_watt'], inplace=True)

# Define features (X)
feature_cols = ['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_watt', 'ervaring', 'rust_sec', 'aantal_intervallen', 'calculated_distance', 'afstand']
X = df[feature_cols]

if wattage == True:
    y = df[['two_k_watt']]
else:
    y = df[['two_k_tijd_sec']]

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# REGRESSOR #################################################################################################
if regressor == True:
    # Train
    regr = DecisionTreeRegressor(max_depth=5)
    regr = regr.fit(X_train, y_train)

    # Predict
    y_val_pred = regr.predict(X_val)
    y_train_pred = regr.predict(X_train)

    if wattage == True:
        # Option 1: convert back to seconds
        y_train_pred = numpy.array([watt_to_pace(x) for x in y_train_pred])
        y_train = numpy.vectorize(watt_to_pace)(y_train)

        y_val_pred = numpy.array([watt_to_pace(x) for x in y_val_pred])
        y_val = numpy.vectorize(watt_to_pace)(y_val)

    # Evaluate
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    print("MSE train reg:", mse_train)
    print("MSE val reg:", mse_val)

# CLASSIFIER ###################################################################################################

if regressor == False: 
    y_train = y_train.multiply(10).astype(int)
    y_val = y_val.multiply(10).astype(int)

    # Train
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    # Predict
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    y_train = y_train * .1
    y_train_pred = y_train_pred * .1
    y_val = y_val * .1
    y_val_pred = y_val_pred * .1
    
    if wattage == True:
        y_val_pred = numpy.array([watt_to_pace(x) for x in y_val_pred])
        y_val = numpy.array([watt_to_pace(x) for x in y_val.iloc[:, 0]])
        y_train_pred = numpy.array([watt_to_pace(x) for x in y_train_pred])
        y_train = numpy.array([watt_to_pace(x) for x in y_train.iloc[:, 0]])

    # Evaluate
    print(y_val)
    print(y_val_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    print("MSE train class:", mse_train)
    mse_test = mean_squared_error(y_val, y_val_pred)
    print("MSE val class:", mse_test)


fig = plt.figure(figsize=(50,15))
_ = tree.plot_tree(regr, 
                   feature_names= feature_cols,  
                #    class_names=df.target_names,
                   filled=True)
plt.show()
