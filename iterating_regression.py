from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy
import random

# Assume you have a DataFrame named 'df' with your data
# If your data is in a CSV file, you can read it using:

df = pd.read_csv('okeanos_processed.csv')

all_cols = df.columns.to_list()

# Specify features for every model you wanna compare.
# Always put to be predicted value as last element in list

model_feat1 = ['500_split_sec', 'two_k_tijd_sec']
model_feat2 = ['500_split_sec', 'two_k_watt']
model_feat3 = ['500_split_watt', 'two_k_tijd_sec']
model_feat4 = ['500_split_watt', 'two_k_watt']
model_feat5 = ['500_split_sec', 'calculated_distance' , 'two_k_tijd_sec']
model_feat6 = ['500_split_sec', 'calculated_distance' , 'man', 'two_k_tijd_sec']
#model_feat7 = ['calculated_distance' , 'mean_500_per_training', 'two_k_tijd_sec']
model_feat7 = ['ervaring', 'man', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'calculated_distance', 'mean_500_per_training', 'two_k_tijd_sec']
model_feat8 = ['ervaring', 'man', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'calculated_distance', 'mean_500_per_training', '500_split_sec', 'two_k_tijd_sec']
model_feat9 = ['ervaring', 'man', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'calculated_distance', 'mean_500_per_training', '500_split_sec', 'aantal_intervallen', 'two_k_tijd_sec']
model_feat10 = ['ervaring', 'man', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'calculated_distance', 'mean_500_per_training', '500_split_sec', 'two_k_tijd_sec']

#model_feat3 = ['mean_watt_per_training', 'two_k_tijd_sec']
#model_feat4 = ['mean_500_per_training', 'two_k_tijd_sec']
#model_feat5 = ['ervaring', 'mean_500_per_training', 'two_k_tijd_sec']
#model_feat6 = ['ervaring', 'man', 'mean_500_per_training', 'two_k_tijd_sec']
#model_feat7 = ['ervaring', 'man', 'mean_500_per_training', 'calculated_distance', 'two_k_tijd_sec']
#model_feat8 = ['ervaring', 'man', 'mean_500_per_training', 'aantal_intervallen', 'two_k_tijd_sec']
#model_feat9 = ['ervaring', 'man', 'mean_500_per_training', 'interval_nummer_1', 'interval_nummer_2', 'interval_nummer_3',
#               'interval_nummer_4', 'interval_nummer_5', 'interval_nummer_6', 'interval_nummer_7', 'interval_nummer_8', 
#               'interval_nummer_9', 'interval_nummer_avg', 'two_k_tijd_sec']
#model_feat7 = ['ervaring', 'man', 'mean_500_per_training', 'rust_sec', 'two_k_tijd_sec']
#model_feat8 = ['ervaring', 'man', 'days_until_2k', 'mean_500_per_training', 'two_k_tijd_sec']
#model_feat9 = ['ervaring', 'man', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'mean_500_per_training', 'two_k_tijd_sec']
#model_feat10 = ['ervaring', 'man', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'interval_afstand', 'mean_500_per_training', 'two_k_tijd_sec']

#model_feat3 = ['man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_500_per_training', 'two_k_tijd_sec']
#model_feat5 = ['man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_watt_per_training', 'two_k_tijd_sec']

performance_list = []

#for iter in range(1, 101):
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
                       model_feat10]:# model_feat8, model_feat9]:

        #print("\n\nModel nr. ", model_nr)
        # Drop all rows if any of the feature data is missing:
        model_feat_df = df.dropna(how = 'any', subset=model_feat, inplace=False)

        # Define features (X) and target variable (y)
        X = model_feat_df[model_feat[:-1]]
        y = model_feat_df[model_feat[-1]]

        # Split the data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=rand_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rand_seed)

        def watt_to_pace(watt):
            if isinstance(watt, float):
                return float(2000 * (float(2.8/float(watt))**(1/3)))



        # Create and train the Multiple Linear Regression model
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_val_pred = linear_reg_model.predict(X_val)
        #print(y_val_pred)

        # validation MSE from wattage to seconds if predicted in watt
        if model_feat[-1] == 'two_k_watt':
            y_val_pred_sec = numpy.array([watt_to_pace(x) for x in y_val_pred])
            y_val_sec = numpy.array([watt_to_pace(x) for x in y_val])
            mse_val_sec = mean_squared_error(y_val_sec, y_val_pred_sec)
            mae_val_sec = (sum(abs(y_val_sec - y_val_pred_sec)) / len(abs(y_val_sec - y_val_pred_sec)))

        # Evaluate the model on the validation set. 
        mse_val = mean_squared_error(y_val, y_val_pred)
        # Absolute afstand
        mae_val = (sum(abs(y_val - y_val_pred)) / len(abs(y_val - y_val_pred)))
        
        #print(mae_val)
        #mse_val_sec = mean_squared_error(y_val_sec, y_val_pred_sec)
        #print(f'Mean Squared Error on Validation Set: {mse_val}')

        # Make predictions on the test set
        y_test_pred = linear_reg_model.predict(X_test)

        # test MSE from wattage to seconds if predicted in watt
        if model_feat[-1] == 'two_k_watt':
            y_test_pred_sec = numpy.array([watt_to_pace(x) for x in y_test_pred])
            y_test_sec = numpy.array([watt_to_pace(x) for x in y_test])
            mse_test_sec = mean_squared_error(y_test_sec, y_test_pred_sec)
            mae_test_sec = (sum(abs(y_test_sec - y_test_pred_sec)) / len(abs(y_test_sec - y_test_pred_sec)))

        # test MSE if prediction was in seconds
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = (sum(abs(y_test - y_test_pred)) / len(abs(y_test - y_test_pred)))
        # Calculate the baseline mean 2k time for y_train 

        # If prediction in watt, calculate seconds for the predicted wattage
        if model_feat[-1] == 'two_k_watt':
            y_train_sec = numpy.array([watt_to_pace(x) for x in y_train])
            baseline_prediction_1 = (sum(y_train_sec))/len(y_train)
            baseline_mse_val_sec = mean_squared_error(numpy.array([baseline_prediction_1 for i in range(len(y_val_pred_sec))]), y_val_sec)
            baseline_mse_test_sec = mean_squared_error(numpy.array([baseline_prediction_1 for i in range(len(y_test_pred_sec))]), y_test_sec)
            baseline_mae_val_sec = ((sum(abs([baseline_prediction_1 for i in range(len(y_val_pred_sec))] - y_val_sec))) / 
                                    len((abs([baseline_prediction_1 for i in range(len(y_val_pred_sec))] - y_val_sec))))
            baseline_mae_test_sec = ((sum(abs([baseline_prediction_1 for i in range(len(y_test_pred_sec))] - y_test_sec))) / 
                                    len((abs([baseline_prediction_1 for i in range(len(y_test_pred_sec))] - y_test_sec))))
        else:
            baseline_prediction = sum(y_train)/len(y_train)
            baseline_mse_val = mean_squared_error(numpy.array([baseline_prediction for i in range(len(y_val))]), y_val)
            baseline_mse_test = mean_squared_error(numpy.array([baseline_prediction for i in range(len(y_test))]), y_test)
            baseline_mae_val = ((sum(abs([baseline_prediction for i in range(len(y_val))] - y_val))) / 
                                    len((abs([baseline_prediction for i in range(len(y_val))] - y_val))))
            baseline_mae_test = ((sum(abs([baseline_prediction for i in range(len(y_test))] - y_test))) / 
                                    len((abs([baseline_prediction for i in range(len(y_test))] - y_test))))
        #print('Dit is model_nr ', model_nr)
        if iter == 1:
            if model_feat[-1] == 'two_k_watt':
                performance_list.append([[mse_val_sec], [baseline_mse_val_sec], [mse_test_sec], [baseline_mse_test_sec],
                                        [len(model_feat_df), model_feat], model_nr, [[mae_val_sec], [baseline_mae_val_sec], 
                                                                                      [mae_test_sec], [baseline_mae_test_sec]]])
            else:
                performance_list.append([[mse_val], [baseline_mse_val], [mse_test], [baseline_mse_test], 
                                         [len(model_feat_df), model_feat], model_nr, [[mae_val],  [baseline_mae_val], 
                                                                                       [mae_test], [baseline_mse_test]]])
            print("Iter == 1 nu is performance list dit:\n", performance_list)
        else:
            if model_feat[-1] == 'two_k_watt':
                performance_list[model_nr - 1][0].append(mse_val_sec)
                performance_list[model_nr - 1][2].append(mse_test_sec)
                performance_list[model_nr - 1][1].append(baseline_mse_val_sec)
                performance_list[model_nr - 1][3].append(baseline_mse_test_sec)
                performance_list[model_nr - 1][6][0].append(mae_val_sec)
                performance_list[model_nr - 1][6][1].append(baseline_mae_val_sec)
                performance_list[model_nr - 1][6][2].append(mae_test_sec)
                performance_list[model_nr - 1][6][3].append(baseline_mae_test_sec)
            else:
                performance_list[model_nr - 1][0].append(mse_val)
                performance_list[model_nr - 1][2].append(mse_test)
                performance_list[model_nr - 1][1].append(baseline_mse_val)
                performance_list[model_nr - 1][3].append(baseline_mse_test)
                performance_list[model_nr - 1][6][0].append(mae_val)
                performance_list[model_nr - 1][6][1].append(baseline_mae_val)
                performance_list[model_nr - 1][6][2].append(mae_test)
                performance_list[model_nr - 1][6][3].append(baseline_mae_test)

        # Print the coefficients of the model
        #coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg_model.coef_})
        #print(coefficients)
        #print(performance_dict['2'][1])
        model_nr += 1

print('\n-------------------------')
print('Model training and predicting done.\n')
print("Printing model performances for ", iter, " iterations...\n")

#print("KIJK: \n", performance_list[0][])

for model_list in performance_list:
    print("Model nr : ", model_list[5])
    print("Features : ", model_list[4][1][:-1])
    print("Amount of rows remaining after dropna : ", model_list[4][0])
    if model_list[4][1][-1] == 'two_k_watt':
        print(f'The model predicts in wattage but the MSE is calculated in seconds')
    else:
        print(f'The model predicts in seconds and the MSE is also calculated in seconds')
    print("MSE valid          :", sum(model_list[0])/len(model_list[0]))
    print("MAE valid          :", sum(model_list[6][0])/len(model_list[6][0]))
    print("Baseline MSE valid :", sum(model_list[1])/len(model_list[1]))
    print("Baseline MAE valid :", sum(model_list[6][1])/len(model_list[6][1]))
    print("MSE test           :", sum(model_list[2])/len(model_list[2]))
    print("MAE test           :", sum(model_list[6][2])/len(model_list[6][2]))
    print("Baseline MSE test  :", sum(model_list[3])/len(model_list[3]))
    print("Baseline MAE test  :", sum(model_list[6][3])/len(model_list[6][3]))
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