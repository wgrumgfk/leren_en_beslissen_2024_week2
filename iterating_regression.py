from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy
import random
from statistics import mean

# Assume you have a DataFrame named 'df' with your data
# If your data is in a CSV file, you can read it using:

df = pd.read_csv('okeanos_processed.csv')

all_cols = df.columns.to_list()

# Initialize a dictionary to store the best cross-validated MSE for each model
best_cv_scores = {f"Model_{i}": float('inf') for i in range(1, 13)}
#number of K-folds for each
num_folds = 6

# Specify features for every model you wanna compare.
# Always put to be predicted value as last element in list
single_model_feat1 = ['500_split_sec', 'two_k_tijd_sec']
single_model_feat2 = ['500_split_watt', 'two_k_watt']
single_model_feat3 = ['mean_500_per_training', 'two_k_tijd_sec']
single_model_feat4 = ['mean_watt_per_training', 'two_k_watt']
#-------------------------------------------------------------
model_feat1 = ['mean_500_per_training', 'two_k_tijd_sec']
model_feat2 = ['mean_500_per_training', 'man', 'two_k_tijd_sec']
model_feat3 = ['mean_500_per_training', 'man', 'ervaring', 'two_k_tijd_sec']
model_feat4 = ['mean_500_per_training', 'man', 'ervaring', 'zwaar', 'two_k_tijd_sec']
model_feat5 = ['mean_500_per_training', 'man', 'ervaring', 'zwaar', 'days_until_2k', 'two_k_tijd_sec']
model_feat6 = ['mean_500_per_training', 'man', 'ervaring', 'zwaar', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'two_k_tijd_sec']
model_feat7 = ['mean_500_per_training', 'man', 'ervaring', 'zwaar', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'calculated_distance', 'two_k_tijd_sec']
model_feat8 = ['mean_500_per_training', 'man', 'ervaring', 'zwaar', 'days_until_2k', 'AT', 'I', 'ID', 'ED', 'calculated_distance', '500_split_sec', 'mean_watt_per_training','two_k_tijd_sec']
#-----------------------------------------------------------------
performance_list = []
acc_thresholds = [1, 2.5, 5, 7.5, 10, 12.5, 15]
iterations = 101

for iter in range(1, iterations + 1):
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
    for model_feat in [single_model_feat1, single_model_feat2,
                       single_model_feat3, single_model_feat4,
                       model_feat1, model_feat2, model_feat3, 
                       model_feat4, model_feat5, model_feat6,
                       model_feat7, model_feat8]:

        #print("\n\nModel nr. ", model_nr)
        # Drop all rows if any of the feature data is missing:
        model_feat_df = df.dropna(how = 'any', subset=model_feat, inplace=False)

        # Define features (X) and target variable (y)
        X = model_feat_df[model_feat[:-1]]
       
        y = model_feat_df[model_feat[-1]]

        # Split the data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=rand_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rand_seed)

        #watt to pace in seconds
        def watt_to_pace(watt):
            if isinstance(watt, float):
                return float(2000 * (float(2.8/float(watt))**(1/3)))

        # Create and train the Multiple Linear Regression model
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_val_pred = linear_reg_model.predict(X_val)
        #print(y_val_pred)

################################################################################################### Cross-validation
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=rand_seed)

        # Convert the target variable to seconds for cross-validation
        if model_feat[-1] == 'two_k_watt':
            y_sec = numpy.array([watt_to_pace(x) for x in y])
        else:
            y_sec = y

        # Calculate cross-validated scores on the converted target variable
        cv_scores_sec = cross_val_score(linear_reg_model, X, y_sec, cv=kf, scoring='neg_mean_squared_error')
        cv_scores_sec = cv_scores_sec * -1

        # Compute the mean of the cross-validated scores
        mean_cv_score_sec = numpy.mean(cv_scores_sec)

        # Update the best cross-validated MSE for each model
        model_name = f"Model_{model_nr}"
        if mean_cv_score_sec < best_cv_scores[model_name]:
            best_cv_scores[model_name] = mean_cv_score_sec
###########################################################################################


        # validation MSE from wattage to seconds if predicted in watt
        if model_feat[-1] == 'two_k_watt':
            y_val_pred_sec = numpy.array([watt_to_pace(x) for x in y_val_pred])
            y_val_sec = numpy.array([watt_to_pace(x) for x in y_val])
            residuals_val_sec = y_val_sec - y_val_pred_sec
            mse_val_sec = mean_squared_error(y_val_sec, y_val_pred_sec)
            mae_val_sec = (sum(abs(residuals_val_sec)) / len(abs(residuals_val_sec)))
            val_acc_sec = []
            for threshold in acc_thresholds:
                val_acc_sec.append(mean([1 if abs(residual) < threshold else 0 for residual in residuals_val_sec]))

        # Evaluate the model on the validation set. 
        residuals_val = y_val - y_val_pred
        mse_val = mean_squared_error(y_val, y_val_pred)
        mae_val = (sum(abs(residuals_val)) / len(abs(residuals_val)))
        val_acc = []
        for threshold in acc_thresholds:
            val_acc.append(mean([1 if abs(residual) < threshold else 0 for residual in residuals_val]))
        
        #print(mae_val)
        #mse_val_sec = mean_squared_error(y_val_sec, y_val_pred_sec)
        #print(f'Mean Squared Error on Validation Set: {mse_val}')

        # Make predictions on the test set
        y_test_pred = linear_reg_model.predict(X_test)

        # test MSE from wattage to seconds if predicted in watt
        if model_feat[-1] == 'two_k_watt':
            y_test_pred_sec = numpy.array([watt_to_pace(x) for x in y_test_pred])
            y_test_sec = numpy.array([watt_to_pace(x) for x in y_test])
            residuals_test_sec = y_test_sec - y_test_pred_sec
            mse_test_sec = mean_squared_error(y_test_sec, y_test_pred_sec)
            mae_test_sec = (sum(abs(residuals_test_sec)) / len(abs(residuals_test_sec)))
            test_acc_sec = []
            for threshold in acc_thresholds:
                test_acc_sec.append(mean([1 if abs(residual) < threshold else 0 for residual in residuals_test_sec]))

        # test MSE if prediction was in seconds
        residuals_test = y_test - y_test_pred
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = (sum(abs(residuals_test)) / len(abs(residuals_test)))
        test_acc = []
        for threshold in acc_thresholds:
            test_acc.append(mean([1 if abs(residual) < threshold else 0 for residual in residuals_test]))

        # If prediction in watt, calculate seconds for the predicted wattage
        # Calculate the baseline mean 2k time for y_train 
        if model_feat[-1] == 'two_k_watt':
            y_train_sec = numpy.array([watt_to_pace(x) for x in y_train])
            baseline_prediction_1 = (sum(y_train_sec))/len(y_train)
            # MSE
            baseline_mse_val_sec = mean_squared_error(numpy.array([baseline_prediction_1 for i in range(len(y_val_pred_sec))]), y_val_sec)
            baseline_mse_test_sec = mean_squared_error(numpy.array([baseline_prediction_1 for i in range(len(y_test_pred_sec))]), y_test_sec)
            # MAE
            baseline_residuals_val_sec = [baseline_prediction_1 for i in range(len(y_val_pred_sec))] - y_val_sec
            baseline_mae_val_sec = mean(abs(baseline_residuals_val_sec))
            baseline_residuals_test_sec = [baseline_prediction_1 for i in range(len(y_test_pred_sec))] - y_test_sec
            baseline_mae_test_sec = mean(abs(baseline_residuals_test_sec))
            # Accuracy
            baseline_val_acc_sec = []
            for threshold in acc_thresholds:
                baseline_val_acc_sec.append(mean([1 if abs(residual) < threshold else 0 for residual in baseline_residuals_val_sec]))
            
            baseline_test_acc_sec = []
            for threshold in acc_thresholds:
                baseline_test_acc_sec.append(mean([1 if abs(residual) < threshold else 0 for residual in baseline_residuals_test_sec]))
        else:
            baseline_prediction = sum(y_train)/len(y_train)
            # MSE
            baseline_mse_val = mean_squared_error(numpy.array([baseline_prediction for i in range(len(y_val))]), y_val)
            baseline_mse_test = mean_squared_error(numpy.array([baseline_prediction for i in range(len(y_test))]), y_test)
            # MAE 
            baseline_residuals_val = [baseline_prediction for i in range(len(y_val_pred))] - y_val
            baseline_mae_val = mean(abs(baseline_residuals_val))
            baseline_residuals_test = [baseline_prediction for i in range(len(y_test_pred))] - y_test
            baseline_mae_test = mean(abs(baseline_residuals_test))
            # Accuracy
            baseline_val_acc = []
            for threshold in acc_thresholds:
                baseline_val_acc.append(mean([1 if abs(residual) < threshold else 0 for residual in baseline_residuals_val]))

            baseline_test_acc = []
            for threshold in acc_thresholds:
                baseline_test_acc.append(mean([1 if abs(residual) < threshold else 0 for residual in baseline_residuals_test]))

        #print('Dit is model_nr ', model_nr)
        if iter == 1:
            if model_feat[-1] == 'two_k_watt':
                performance_list.append([[mse_val_sec], [baseline_mse_val_sec], [mse_test_sec], [baseline_mse_test_sec],
                                        [len(model_feat_df), model_feat], model_nr, [[mae_val_sec], [baseline_mae_val_sec], 
                                                                                      [mae_test_sec], [baseline_mae_test_sec]],
                                                                                      [val_acc_sec], [baseline_val_acc_sec], [test_acc_sec], [baseline_test_acc_sec]])
            else:
                performance_list.append([[mse_val], [baseline_mse_val], [mse_test], [baseline_mse_test], 
                                         [len(model_feat_df), model_feat], model_nr, [[mae_val],  [baseline_mae_val], 
                                                                                       [mae_test], [baseline_mse_test]],
                                                                                       [val_acc], [baseline_val_acc], [test_acc], [baseline_test_acc]])
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
                performance_list[model_nr - 1][7].append(val_acc_sec)
                performance_list[model_nr - 1][8].append(baseline_val_acc_sec)
                performance_list[model_nr - 1][9].append(test_acc_sec)
                performance_list[model_nr - 1][10].append(baseline_test_acc_sec)
            else:
                performance_list[model_nr - 1][0].append(mse_val)
                performance_list[model_nr - 1][2].append(mse_test)
                performance_list[model_nr - 1][1].append(baseline_mse_val)
                performance_list[model_nr - 1][3].append(baseline_mse_test)
                performance_list[model_nr - 1][6][0].append(mae_val)
                performance_list[model_nr - 1][6][1].append(baseline_mae_val)
                performance_list[model_nr - 1][6][2].append(mae_test)
                performance_list[model_nr - 1][6][3].append(baseline_mae_test)
                performance_list[model_nr - 1][7].append(val_acc)
                performance_list[model_nr - 1][8].append(baseline_val_acc)
                performance_list[model_nr - 1][9].append(test_acc)
                performance_list[model_nr - 1][10].append(baseline_test_acc)

        # Print the coefficients of the model
        #coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg_model.coef_})
        #print(coefficients)

        model_nr += 1

# print('\n-------------------------')
# print('Model training and predicting done.\n')
# print("Printing model performances for ", iter, " iterations...\n")

# #print("KIJK: \n", performance_list[0][])

# for model_list in performance_list:
#     print('--------------------------')
#     print("Model nr : ", model_list[5])
#     print("Features : ", model_list[4][1][:-1])
#     print("Amount of rows remaining after dropna : ", model_list[4][0])
#     if model_list[4][1][-1] == 'two_k_watt':
#         print(f'The model predicts in wattage but the MSE is calculated in seconds')
#     else:
#         print(f'The model predicts in seconds and the MSE is also calculated in seconds')
#     print('\nVALIDATION set:')
#     print("MSE valid          :", sum(model_list[0])/len(model_list[0]))
#     print("MAE valid          :", sum(model_list[6][0])/len(model_list[6][0]))
#     for i in range(len(acc_thresholds)):
#         temp_acc = 0
#         for j in range(iterations):
#             temp_acc += model_list[7][j][i]
        
#         print("Accuracy valid within range ", acc_thresholds[i], " seconds: ", temp_acc/iterations)
    
#     print('\nBASELINE VALIDATION:')
#     print("Baseline MSE valid :", sum(model_list[1])/len(model_list[1]))
#     print("Baseline MAE valid :", sum(model_list[6][1])/len(model_list[6][1]))
#     for i in range(len(acc_thresholds)):
#         temp_acc = 0
#         for j in range(iterations):
#             temp_acc += model_list[8][j][i]
        
#         print("Accuracy baseline valid within range ", acc_thresholds[i], " seconds: ", temp_acc/iterations)

#     print('\nTEST set:')
#     print("MSE test           :", sum(model_list[2])/len(model_list[2]))
#     print("MAE test           :", sum(model_list[6][2])/len(model_list[6][2]))
#     for i in range(len(acc_thresholds)):
#         temp_acc = 0
#         for j in range(iterations):
#             temp_acc += model_list[9][j][i]
        
#         print("Accuracy test within range ", acc_thresholds[i], "seconds: ", temp_acc/iterations)

#     print('\nBASELINE TEST:')
#     print("Baseline MSE test  :", sum(model_list[3])/len(model_list[3]))
#     print("Baseline MAE test  :", sum(model_list[6][3])/len(model_list[6][3]))
#     for i in range(len(acc_thresholds)):
#         temp_acc = 0
#         for j in range(iterations):
#             temp_acc += model_list[10][j][i]
        
#         print("Accuracy baseline test within range ", acc_thresholds[i], "seconds: ", temp_acc/iterations)
    
#     print("\n")


for model_name, best_cv_score in best_cv_scores.items():
    print(f"{model_name}: Best Cross-validated MSE = {best_cv_score}")


"""


    plt.hist(residuals_val_sec, bins=20)
    plt.title("Histogram of Residuals (Validation Set)")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()
"""