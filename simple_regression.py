from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import copy
import math

# Assume you have a DataFrame named 'df' with your data
# If your data is in a CSV file, you can read it using:

df = pd.read_csv('okeanos_processed.csv')

all_cols = df.columns.to_list()

# model_feat1 and model_feat2 are identical apart from extra dummy variable. As you can see the MSE is the same for both models
# THis shows you can leave out 1 column for every catagorical variable you are transforming into dummy variables.
model_feat1 = ['days_until_2k', 'man', 'vrouw', 'zwaar', 'licht', 'AT', 'I', 'ID', 'ED', 'ED+', 'ervaring', 'mean_500_per_training', 'two_k_tijd_sec', 'interval_afstand']
model_feat2 = ['days_until_2k', 'man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_500_per_training', 'two_k_tijd_sec']

# Specify features for every model you wanna compare.
model_feat3 = ['man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_500_per_training', 'two_k_tijd_sec']
model_feat4 = ['man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_watt_per_training', 'two_k_watt']
model_feat5 = ['man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_500_per_training', 'two_k_watt']
model_feat6 = ['man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_watt_per_training', 'two_k_tijd_sec']
model_feat7 = ['man', 'zwaar', 'AT', 'I', 'ID', 'ED', 'ervaring', 'mean_watt_per_training', 'two_k_watt', 'rust_sec']

# Initialize mse and model variables.
model_nr = 1
best_val_mse = 9999999999
best_test_mse = 9999999999
best_model_nr = 0

# For every model calculate the val_mse. Store the model with best val_mse.
# Also print the test_mse for this model.
for model_feat in [model_feat1, model_feat2, model_feat3, 
                   model_feat4, model_feat5, model_feat6, 
                   model_feat7]:
    
    print("\n\nModel nr. ", model_nr)
    # Drop all rows if any of the feature data is missing:
    model_feat_df = df.dropna(how = 'any', subset=model_feat, inplace=False)

    # Define features (X) and target variable (y)
    # Either predict in seconden or watt.
    if 'two_k_tijd_sec' in model_feat:
        model_feat.remove('two_k_tijd_sec')
        y = model_feat_df['two_k_tijd_sec']
    else:
        model_feat.remove('two_k_watt')
        y = model_feat_df['two_k_watt']

    X = model_feat_df[model_feat]

    print("\nFeatures:\n", model_feat)
    print('\n\nDeze specifieke subset van kolommen heeft ', len(model_feat_df), " rijen waarbij alle kolommen zijn ingevuld.\n")

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=40)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=40)

    # Create and train the Multiple Linear Regression model
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = linear_reg_model.predict(X_val)
    #print(y_val_pred)

    # Evaluate the model on the validation set. Calculate the 
    # MSE in seconds, so calculate back if prediction was in watt.
    if 'two_k_tijd_sec' in model_feat:
        mse_val = mean_squared_error(y_val, y_val_pred)
    else:
        mse_val = float(2000 * float(2.8/float(mean_squared_error(y_val, y_val_pred)))**(1/3))

    print(f'Mean Squared Error on Validation Set: {mse_val}')
    print(f'Mean error on Validation Set: {math.sqrt(mse_val)}\n')

    # Make predictions on the test set
    y_test_pred = linear_reg_model.predict(X_test)

    # Evaluate the model on the test set
    # Again predict back to seconds if prediction was in watt.
    if 'two_k_tijd_sec' in model_feat:
        mse_test = mean_squared_error(y_test, y_test_pred)
    else:
        mse_test = float(2000 * float(2.8/float(mean_squared_error(y_test, y_test_pred)))**(1/3))

    print(f'Mean Squared Error on Test Set: {mse_test}')
    print(f'Mean error on Test Set: {math.sqrt(mse_test)}\n')

    if mse_val < best_val_mse:
        best_val_mse = mse_val
        best_test_mse = mse_test
        best_model_nr = model_nr

    # Print the coefficients of the model
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg_model.coef_})
    print(coefficients)

    model_nr += 1

# Print the best model and its mse's to screen.
print("\nRESULTS:\nBest model nr: ", best_model_nr)
print("validation MSE: ", best_val_mse)
print("test MSE: ", best_test_mse)