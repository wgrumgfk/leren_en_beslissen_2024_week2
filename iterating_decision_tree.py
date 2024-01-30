from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import random

df = pd.read_csv('okeanos_processed.csv')

all_cols = df.columns.to_list()


def watt_to_pace(watt):
    if isinstance(watt, float):
        return float(2000 * (float(2.8/float(watt))**(1/3)))
    
performance_dict = dict()


for iter in range(1, 101):
    rand_seed = random.randint(0, 100000)

    best_depth = None
    min_val_mse = float('inf')
    best_residuals = None
    best_iteration = None

    # For every model calculate the val_mse. Store the model with best val_mse.
    # Also print the test_mse for this model.
    for depth in (range(1,15)):
        model_feat= ['days_until_2k','man','zwaar','AT','I','ID','ED','rust_sec','afstand','ervaring','500_split_watt','aantal_intervallen', 'calculated_distance', 'two_k_watt']
        
        # Drop all rows if any of the feature data is missing:
        model_feat_df = df.dropna(how = 'any', subset=model_feat, inplace=False)

        # Define features (X) and target variable (y)
        X = model_feat_df[model_feat[:-1]]
        y = model_feat_df[model_feat[-1]]

        # Split the data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=rand_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rand_seed)

        # Train
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train, y_train)

        # Predict
        y_val_pred = regr.predict(X_val)
        y_train_pred = regr.predict(X_train)

        baseline_prediction = sum(y_train)/len(y_train)

        # Evaluate validation
        y_val_pred = numpy.array([watt_to_pace(x) for x in y_val_pred])
        y_val = numpy.array([watt_to_pace(x)  for x in y_val])
        mse_val = mean_squared_error(y_val, y_val_pred)
        baseline_mse_val = mean_squared_error(numpy.array([watt_to_pace(baseline_prediction) for _ in range(len(y_val))])  , y_val)

        # Evaluate training
        y_train_pred = numpy.array([watt_to_pace(x) for x in y_train_pred])
        y_train = numpy.array([watt_to_pace(x)  for x in y_train])
        mse_train = mean_squared_error(y_train, y_train_pred)
        baseline_mse_train = mean_squared_error(numpy.array([watt_to_pace(baseline_prediction)  for _ in range(len(y_train))]), y_train)

        if depth == 7:  # Assuming you are interested in depth 7
            residuals_val_sec = y_val - y_val_pred
            if mse_val < min_val_mse:
                min_val_mse = mse_val
                best_depth = depth
                best_residuals = residuals_val_sec
                best_iteration = iter
        
      
        if iter == 1:
            performance_dict[depth] = [[mse_val], [baseline_mse_val], [mse_val], [baseline_mse_val], [len(model_feat_df), model_feat]]
        else:
            if model_feat[-1] == 'two_k_watt':
                performance_dict[depth][0].append(mse_val)
                performance_dict[depth][2].append(mse_train)
            performance_dict[depth][1].append(baseline_mse_val)
            performance_dict[depth][3].append(baseline_mse_train)

        # Print the coefficients of the model
        #coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg_model.coef_})
        #print(coefficients)

if best_residuals is not None:
    plt.hist(best_residuals, bins=20)
    plt.title(f"Residuals (Validation Set) for the best model (Depth {best_depth}),Iteration {best_iteration}")
    plt.xlabel("Residuals")
    plt.ylabel("Percantage")
    plt.show()
print('\n-------------------------')
print('Model training and predicting done.\n')
print("Printing model performances for ", iter, " iterations...\n")

for k,v in performance_dict.items():
    print("Depth :", k)
    print("Features : ", v[4][1][:-1])
    print("Amount of rows remaining after dropna : ", v[4][0])
    if v[4][1][-1] == 'two_k_watt':
        print(f'The model predicts in wattage but the MSE is calculated in seconds')
    else:
        print(f'The model predicts in seconds and the MSE is also calculated in seconds')
    print("MSE valid          :", sum(v[0])/len(v[0]))
    print("Baseline MSE valid :", sum(v[1])/len(v[1]))
    print("MSE train           :", sum(v[2])/len(v[2]))
    print("Baseline MSE train  :", sum(v[3])/len(v[3]))
    print("\n")

"""
# Print the best model and its mse's to screen.
print("\nRESULTS:\nBest model nr: ", best_model_nr)
print("validation MSE: ", best_val_mse)
print("test MSE: ", best_test_mse)
print('----------------------------')
for k,v in performance_dict.items():
    print("\nModel number ", k)
    print("Validation MSE: ", v[0])
    print("Test       MSE: ", v[1])
"""