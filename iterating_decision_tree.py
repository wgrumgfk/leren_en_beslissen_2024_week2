from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from statistics import mean
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
mse_5 = []
mse_5_baseline =[]
vector_list = []
vector_list_baseline = []

model_feat= ['days_until_2k','man','zwaar','AT','I','ID','ED','rust_sec','afstand','ervaring','mean_500_per_training','aantal_intervallen', 'calculated_distance', 'two_k_watt']
importances = {feature: [] for feature in model_feat[:-1]}
best_importance = {feature: [] for feature in model_feat[:-1]}

for iter in range(1, 101):
    rand_seed = random.randint(0, 100000)

    best_depth = None
    min_val_mse = float('inf')
    min_test_mse = float('inf')
    best_residuals = None
    best_iteration = None

    # For every model calculate the val_mse. Store the model with best val_mse.
    # Also print the test_mse for this model.
    for depth in (range(1,15)):
        
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
        y_test_pred = regr.predict(X_test)
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

        # Evaluate test
        y_test_pred = numpy.array([watt_to_pace(x) for x in y_test_pred])
        y_test = numpy.array([watt_to_pace(x)  for x in y_test])
        mse_test = mean_squared_error(y_test, y_test_pred)
        baseline_mse_test = mean_squared_error(numpy.array([watt_to_pace(baseline_prediction) for _ in range(len(y_test))]), y_test)
        
        if depth == 5:
            mse_5.append(mse_test)
            mse_5_baseline.append(baseline_mse_test)
            residuals_test_sec = y_test - y_test_pred
            residuals_test_baseline = y_test - numpy.array([watt_to_pace(baseline_prediction) for _ in range(len(y_test))])
            vector_list.append(residuals_test_sec)
            vector_list_baseline.append(residuals_test_baseline)
            feature_importances = regr.feature_importances_
            sorted_indices = numpy.argsort(feature_importances)[::-1]
            for index in sorted_indices:
                importances[X.columns[index]].append(feature_importances[index])
            
            if mse_test < min_test_mse:
                best_model = regr
                best_feature_importance = regr.feature_importances_
                sorted_indices = numpy.argsort(best_feature_importance)[::-1]
                for index in sorted_indices:
                    best_importance[X.columns[index]] = best_feature_importance[index]
      
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

residuals = numpy.concatenate(vector_list, axis=None)
print('mae', mean(abs(residuals)))

accuracies = []
for threshold in([1,2.5,5,7.5,10,12.5,15,17.7,20]):
    succes = 0
    for residual in residuals:
        if abs(residual) <= threshold:
            succes += 1
    accuracies.append(succes / len(residuals))

print(accuracies)

plt.hist(residuals, weights=numpy.ones(len(residuals)) / len(residuals) * 100, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Percentage")
plt.show()

residuals_baseline = numpy.concatenate(vector_list_baseline, axis=None)
print('mae baseline', mean(abs(residuals_baseline)))

accuracies_baseline = []
for threshold in([1,2.5,5,7.5,10,12.5,15,17.7,20]):
    succes = 0
    for residual in residuals_baseline:
        if abs(residual) <= threshold:
            succes += 1
    accuracies_baseline.append(succes / len(residuals_baseline))

print(accuracies_baseline)

plt.hist(residuals_baseline, weights=numpy.ones(len(residuals_baseline)) / len(residuals_baseline) * 100, bins = 20)
plt.xlabel("Residuals")
plt.ylabel("Percentage")
plt.show()


print(mean(mse_5))
print(mean(mse_5_baseline))
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

mean_importances = {feature: numpy.mean(importance) for feature, importance in importances.items()}
percentage_importances = {feature: (sum(importance > 0 for importance in importancess) / len(importancess)) * 100 if len(importancess) > 0 else 0 for feature, importancess in importances.items()}
print(mean_importances)
print(percentage_importances)
print(best_importance)

fig = plt.figure(figsize=(50,15))
_ = tree.plot_tree(best_model, 
                   feature_names = model_feat,
                   filled=True)
plt.show()
