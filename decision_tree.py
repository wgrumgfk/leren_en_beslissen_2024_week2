from sklearn import tree
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Make sure all rows are complete
df = pd.read_csv('okeanos_processed_decision_tree.csv')
df.dropna(how = 'any', subset=['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','aantal_intervallen','intervaltype','interval_afstand','interval_nummer', 'two_k_tijd_sec'], inplace=True)

# Define features (X) and target variable (y)
X = df[['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_sec','interval_afstand']]
y = df[['two_k_tijd_sec']]

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_val_pred = clf.predict(X_val)
print(y_val_pred)

# Evaluate performance
mse_val = mean_squared_error(y_val, y_val_pred)
print(f'Mean Squared Error on Validation Set: {mse_val}')