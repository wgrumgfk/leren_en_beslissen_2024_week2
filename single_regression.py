from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy

# Assume you have a DataFrame named 'df' with your data
# If your data is in a CSV file, you can read it using:

df = pd.read_csv('okeanos_processed.csv')

all_cols = df.columns.to_list()

# Specify features for every model you wanna compare.
# Always put to be predicted value as last element in list
model_feat1 = ['days_until_2k', 'man', 'vrouw', 'zwaar', 'licht', 'AT', 'I', 'ID', 'ED', 'ED+', 'ervaring', 'mean_500_per_training', 'two_k_tijd_sec']
model_feat2 = ['days_until_2k', 'man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_500_per_training', 'two_k_tijd_sec']
model_feat8 = ['ervaring', 'man', 'days_until_2k', 'interval_afstand', 'mean_500_per_training', 'two_k_tijd_sec']
#rand_seed = random.randint(0, 10000)
rand_seed = 7
model_nr = 1
print('------------------------------')
print('Random seed is ', rand_seed)
print('\n')
# For every model calculate the val_mse. Store the model with best val_mse.
# Also print the test_mse for this model.
for model_feat in [model_feat1, model_feat2, model_feat8]:
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

    # Calculate the baseline mean 2k time for y_train
    baseline_prediction = sum(y_train)/len(y_train)

    # Create and train the Multiple Linear Regression model
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = linear_reg_model.predict(X_val)
    #print(y_val_pred)

    # Evaluate the model on the validation set. Calculate the 
    # MSE in seconds, so calculate back if prediction was in watt.
    mse_val = mean_squared_error(y_val, y_val_pred)
    baseline_mse_val = mean_squared_error(numpy.array([baseline_prediction for i in range(len(y_val_pred))]), y_val)

    print(f'Mean Squared Error on Validation Set: {mse_val}')
    print(f'Baseline MSE on Validation Set:       {baseline_mse_val}')
    #print(f'Mean error on Validation Set: {math.sqrt(mse_val)}\n')

    # Make predictions on the test set
    y_test_pred = linear_reg_model.predict(X_test)

    # Evaluate the model on the test set
    # Again predict back to seconds if prediction was in watt.
    mse_test = mean_squared_error(y_test, y_test_pred)
    baseline_mse_test = mean_squared_error(numpy.array([baseline_prediction for i in range(len(y_test_pred))]), y_test)

    print(f'Mean Squared Error on Test Set: {mse_test}')
    print(f'Baseline MSE on Test Set      : {baseline_mse_test}')

    # Print the coefficients of the model
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg_model.coef_})
    print(coefficients)

    model_nr += 1

print('\n-------------------------')