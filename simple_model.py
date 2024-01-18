from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Assume you have a DataFrame named 'df' with your data
# If your data is in a CSV file, you can read it using:

df = pd.read_csv('okeanos_processed.csv')
df.dropna(how = 'any', subset=['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','two_k_watt','500_split_watt','aantal_intervallen','intervaltype','interval_afstand','interval_nummer','2k tijd','mean_watt_per_training'], inplace=True)

# Convert categorical variables to dummy/indicator variables (one-hot encoding)


# Define features (X) and target variable (y)
X = df[['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_watt','interval_afstand','interval_nummer','mean_watt_per_training']]
y = df['two_k_watt']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train the Multiple Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = linear_reg_model.predict(X_val)

print(y_val_pred)
# Evaluate the model on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
print(f'Mean Squared Error on Validation Set: {mse_val}')

# Make predictions on the test set
y_test_pred = linear_reg_model.predict(X_test)

# Evaluate the model on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'Mean Squared Error on Test Set: {mse_test}')

# Print the coefficients of the model
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg_model.coef_})
print(coefficients)