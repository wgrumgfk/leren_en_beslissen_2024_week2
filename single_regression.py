from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import metrics
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statistics import mean

# Assume you have a DataFrame named 'df' with your data
# If your data is in a CSV file, you can read it using:

df = pd.read_csv('okeanos_processed.csv')

all_cols = df.columns.to_list()

num_folds = 5

################################################################################# Defenitions
""" This function turns watagge of 2000 m into seconds"""
def watt_to_pace(watt):
    if isinstance(watt, float):
        return float(2000 * (float(2.8/float(watt))**(1/3)))

# Specify features for every model you wanna compare.
# Always put to be predicted value as last element in list
model_feat1 = ['days_until_2k', 'man', 'vrouw', 'zwaar', 'licht', 'AT', 'I', 'ID', 'ED', 'ED+', 'ervaring', 'mean_watt_per_training', 'two_k_watt']
model_feat2 = ['days_until_2k', 'man', 'vrouw', 'zwaar', 'licht', 'AT', 'I', 'ID', 'ED', 'ED+', 'ervaring', 'mean_watt_per_training', 'two_k_tijd_sec']
model_feat3 = ['ervaring', 'man', 'days_until_2k', 'interval_afstand', 'mean_500_per_training', 'two_k_watt']
#rand_seed = random.randint(0, 10000)
rand_seed = 7
model_nr = 1
print('------------------------------')
print('Random seed is ', rand_seed)
print('\n')
# For every model calculate the val_mse. Store the model with best val_mse.
# Also print the test_mse for this model.
for model_feat in [model_feat1, model_feat2, model_feat3]:
    print("\n\nModel nr. ", model_nr)
    # Drop all rows if any of the feature data is missing:
    model_feat_df = df.dropna(how = 'any', subset=model_feat, inplace=False)

    # Define features (X) and target variable (y)
    X = model_feat_df[model_feat[:-1]]
    y = model_feat_df[model_feat[-1]]

    print("\nFeatures:\n", model_feat)
    print('\n\nDeze specifieke subset van kolommen heeft ', len(model_feat_df), " rijen waarbij alle kolommen zijn ingevuld.\n")


    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=rand_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rand_seed)


########################################################################################################################## Use KFold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=rand_seed)
    

########################################################################################################################## linear model 
    # Create and train the Multiple Linear Regression model
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = linear_reg_model.predict(X_val)
######################################################################################################################### cross-validation
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
######################################################################################################################### printing the calculated cross validation scores
    print("\n\nModel nr. ", model_nr)
    print("Cross-validated MSE: ", mean_cv_score_sec)
##############################################################################################################
    # validation from wattage to seconds
    if model_feat[-1] == 'two_k_watt':
        y_val_pred_sec = numpy.array([watt_to_pace(x) for x in y_val_pred])
        y_val_sec = numpy.array([watt_to_pace(x) for x in y_val])
        mse_val_sec = mean_squared_error(y_val_sec, y_val_pred_sec)

    # Evaluate the model on the validation set. Calculate the 
    # MSE in seconds, so calculate back if prediction was in watt.
    mse_val = mean_squared_error(y_val, y_val_pred)   

    # Make predictions on the test set
    y_test_pred = linear_reg_model.predict(X_test)

    # test predictions from wattage to seconds
    if model_feat[-1] == 'two_k_watt':
        y_test_pred_sec = numpy.array([watt_to_pace(x) for x in y_test_pred])
        y_test_sec = numpy.array([watt_to_pace(x) for x in y_test])
        mse_test_sec = mean_squared_error(y_test_sec, y_test_pred_sec)

    # Evaluate the model on the test set
    # Again predict back to seconds if prediction was in watt.
    mse_test = mean_squared_error(y_test, y_test_pred)

    # transform y_train form wattage to seconds
    if model_feat[-1] == 'two_k_watt':
        y_train_sec = numpy.array([watt_to_pace(x) for x in y_train])
        baseline_prediction_1= (sum(y_train_sec))/len(y_train)
        baseline_mse_val_sec = mean_squared_error(numpy.array([baseline_prediction_1 for i in range(len(y_val_pred_sec))]), y_val_sec)
        baseline_mse_test_sec = mean_squared_error(numpy.array([baseline_prediction_1 for i in range(len(y_test_pred_sec))]), y_test_sec)
    else: 
        # Calculate the baseline mean 2k time for y_train
        baseline_pred = (sum(y_train))/len(y_train)
        baseline_mse_val = mean_squared_error(numpy.array([baseline_pred for i in range(len(y_val_pred))]), y_val)
        baseline_mse_test = mean_squared_error(numpy.array([baseline_pred for i in range(len(y_test_pred))]), y_test)

    r_squared_val_sec = r2_score(y_val_sec, y_val_pred_sec)

    n = len(y_val_sec)
    k = len(model_feat) - 1  # Number of predictors
    adjusted_r_squared_val_sec = 1 - ((1 - r_squared_val_sec) * (n - 1) / (n - k - 1))


    # print statements
    if model_feat[-1] == 'two_k_watt':
        print(f'The model predicts in wattage but the MSE is calculated in seconds')
        print(f'Mean Squared Error on Validation Set with seconds: {mse_val_sec}')
        print(f'Baseline MSE on Validation Set with seconds:       {baseline_mse_val_sec}\n')
        print(f'Mean Squared Error on Test Set with seconds: {mse_test_sec}')
        print(f'Baseline MSE on Test Set with seconds     : {baseline_mse_test_sec}\n')
        print("Mean absolute error validation:",mean_absolute_error(y_val_sec, y_val_pred_sec))
        print("Mean absolute error test:",mean_absolute_error(y_test_sec, y_test_pred_sec))
        print("R-squared :",r_squared_val_sec)
        print("Adjusted R squared val", adjusted_r_squared_val_sec)
        print("R-squared :",r2_score(y_test_sec, y_test_pred_sec))
    else:
        print(f'The model predicts in seconds and the MSE is also calculated in seconds')
        print(f'Mean Squared Error in seconds on Validation Set: {mse_val}')
        print(f'Baseline MSE on Validation Set:       {baseline_mse_val} \n')
        print(f'Mean Squared Error on Test Set: {mse_test}')
        print(f'Baseline MSE on Test Set      : {baseline_mse_test} \n')

    ###################residual analasys
    residuals_val_sec = y_val_sec - y_val_pred_sec
    residuals_test_sec = y_test - y_test_pred
    #####################



########################## acuraccy of model 
    threshold = 10
    ACC = []

    good = 0
    for i in range(len(y_val)):
        if abs(y_test_sec[i] - y_test_pred_sec[i]) < threshold:
            good += 1

    ACC.append(good / len(y_val))
    print('accuracy with threshold of', threshold, ':', mean(ACC))
##########################################


    # Print the coefficients of the model
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg_model.coef_})
    print(coefficients)

  
########################### Histogram of residuals
    plt.hist(residuals_val_sec, bins=20)
    plt.title("Histogram of Residuals (Validation Set)")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()
############################
    model_nr += 1

print('\n-------------------------')