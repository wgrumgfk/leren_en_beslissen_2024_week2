from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy
import random
import pandas as pd

# Assume you have a DataFrame named 'df' with your data
# If your data is in a CSV file, you can read it using:

df = pd.read_csv('okeanos_processed.csv')

all_cols = df.columns.to_list()

# Specify features for every model you wanna compare.
# Always put to be predicted value as last element in list

model_feat1 = ['500_split_watt', 'two_k_tijd_sec']
model_feat2 = ['500_split_sec', 'two_k_tijd_sec']
model_feat3 = ['mean_watt_per_training', 'two_k_tijd_sec']
model_feat4 = ['mean_500_per_training', 'two_k_tijd_sec']
model_feat5 = ['ervaring', 'mean_500_per_training', 'two_k_tijd_sec']
model_feat6 = ['ervaring', 'man', 'mean_500_per_training', 'two_k_tijd_sec']
# model_feat7 = ['ervaring', 'man', 'mean_500_per_training', "5x5'", "3x15'", "3x20'", "3x2000m/5'r", '6000m', '4x1500m', "3x1000m/5'r",
#  "6x500m/2'r", '1500m', "1'", '2000m', "30'", '3x2000m', "3x20'/3'r",
#  "3x6x1'/1'r", 'minuutjes', '2x2000m', '6x500m', '3x1000m', "8x5'/3'r",
#  "3x4000m/4'r", "4x8'/4'r", "3x10'", '1000m', '1500m + 500m', "4x8'", "8x3'",
#  '4x750m', "3x10'/5'r", '2x2000m + 500m', "3x7x1'/1'r /3'r", '1000m + 500m',
#  "3x5x1'/1'r /5'r", "4x5'", "3x12'", "2x25'", '8x500m', "4x2000/5'r", '100m',
#  '500m', "3x20'/5'r", '1500m+500m', "HOP+3x1'r", '3x4000m', "3x7'", "3x8'",
#  '1000m+500m', "3x10'/3'r", "3x2000/3'r", "4x5'/5'r", '2000', "3x12'/3'r",
#  "4x500/5'r", "3x3000/5'r", "3x15'/3'r", "8x3'/3'r", "7x3'/3'r", "2x19'/3'r",
#  "2x3000/5'r", "6x750/5'r", "2x2000/5'r", '1000', "3x8'/5'r", '1500m+750m',
#  "3x12'/5'r", "2x6x1'/1'r", "2x7x1'/1'r", "3x11'", "6x6'", "3x13'", "5x8'", "20'",
#  "3x2000m/4'r", "3x10'/4'r", "3x1000m/3'r", "8x3'/2'r", "6x750/3'r",
#  "3x7x1'/1'r", "3x1500/5'r", "4x8'/5'r", '6000', "2x10'/5'r", "2x4x20''/40''r",
#  "1x20'", "3x5x1'/1'r", "6x5'/2'r", "7x3'/2'r", "3x2000/5'r" ,"4x4x40''/40''r",
#  "9x3'/2'r", "3x8x1'/1'r", "5x5'/3'r", "5x7'/3'r", "5x9'/3'r", "3x1000/3'r",
#  "2x12'/8'r", "3x8x40'/20'r", "3x16'/3'r", "3x7'/5'r", "6x3'/3'r", "9x3'/3'r", 'two_k_tijd_sec']
model_feat7 = ['ervaring', 'man', 'mean_500_per_training', 'rust_sec', 'two_k_tijd_sec']
model_feat8 = ['ervaring', 'man', 'days_until_2k', 'mean_500_per_training', 'two_k_tijd_sec']
model_feat9 = ['ervaring', 'man', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'mean_500_per_training', 'two_k_tijd_sec']
model_feat10 = ['ervaring', 'man', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'interval_afstand', 'mean_500_per_training', 'two_k_tijd_sec']

performance_dict = dict()

for iter in range(1, 101):
    # Initialize mse and model variables.
    rand_seed = random.randint(0, 100000)
    #rand_seed = 7
    model_nr = 1

    print('------------------------------')
    print('Iteration: ', iter)
    print('Random seed is ', rand_seed)
    print('\n')
    # For every model calculate the val_mse. Store the model with best val_mse.
    # Also print the test_mse for this model.
    for model_feat in [model_feat1, model_feat2, model_feat3, 
                       model_feat4, model_feat5, model_feat6,
                       model_feat7, model_feat8, model_feat9,
                       model_feat10]:

        # print("\n\nModel nr. ", model_nr)
        # Drop all rows if any of the feature data is missing:
        model_feat_df = df.dropna(how = 'any', subset=model_feat, inplace=False)

        model_feat_df['two_k_tijd_sec'] = model_feat_df['two_k_tijd_sec'].multiply(10)
        model_feat_df['500_split_sec'] = model_feat_df['500_split_sec'].multiply(10)
        model_feat_df.round({'two_k_tijd_sec': 1})
        model_feat_df['two_k_tijd_sec'] = model_feat_df['two_k_tijd_sec'].astype(int)


        # Define features (X) and target variable (y)
        X = model_feat_df[model_feat[:-1]]
        y = model_feat_df[model_feat[-1]]

        # Split the data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=rand_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rand_seed)

        # Calculate the baseline mean 2k time for y_train
        baseline_prediction = sum(y_train)/len(y_train)

        # Create and train the Multiple Linear Regression model
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        # Make predictions on the validation set
        y_val_pred = clf.predict(X_val)

        # Evaluate the model on the validation set. Calculate the 
        mse_val = mean_squared_error(y_val, y_val_pred)
        baseline_mse_val = mean_squared_error(numpy.array([baseline_prediction for i in range(len(y_val_pred))]), y_val)
        #print(f'Mean Squared Error on Validation Set: {mse_val}')

        # Make predictions on the test set
        y_test_pred = clf.predict(X_test)

        # Evaluate the model on the test set
        mse_test = mean_squared_error(y_test, y_test_pred)
        baseline_mse_test = mean_squared_error(numpy.array([baseline_prediction for i in range(len(y_test_pred))]), y_test)
        #print(f'Mean Squared Error on Test Set: {mse_test}')

        if iter == 1:
            performance_dict[model_nr] = [[mse_val], [baseline_mse_val], [mse_test], [baseline_mse_test], [len(model_feat_df), model_feat[:-1]]]
        else:
            performance_dict[model_nr][0].append(mse_val)
            performance_dict[model_nr][1].append(baseline_mse_val)
            performance_dict[model_nr][2].append(mse_test)
            performance_dict[model_nr][3].append(baseline_mse_test)

        # Print the coefficients of the model
        #coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg_model.coef_})
        #print(coefficients)

        model_nr += 1

print('\n-------------------------')
print('Model training and predicting done.\n')
print("Printing model performances for ", iter, " iterations...\n")

for k,v in performance_dict.items():
    print("Model nr :", k)
    print("Features : ", v[4][1])
    print("Amount of rows remaining after dropna : ", v[4][0])
    print("MSE valid          :", sum(v[0])/len(v[0]))
    print("Baseline MSE valid :", sum(v[1])/len(v[1]))
    print("MSE test           :", sum(v[2])/len(v[2]))
    print("Baseline MSE test  :", sum(v[3])/len(v[3]))
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