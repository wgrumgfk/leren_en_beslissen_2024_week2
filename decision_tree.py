from sklearn import tree
import pandas as pd
import numpy
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import tree

# Make sure all rows are complete
df = pd.read_csv('okeanos_processed.csv')
df.dropna(how = 'any', subset=['two_k_tijd_sec', 'days_until_2k', 'man', 'zwaar','AT','I','ID','ED','aantal_intervallen','afstand','interval_afstand'], inplace=True)

# Make 2k time and 500 split in tenths of seconds
df['two_k_tijd_sec'] = df['two_k_tijd_sec'].multiply(10).astype(int)
# df['500_split_sec'] = df['500_split_sec'].multiply(10)
df.round({'two_k_tijd_sec': 1})

# Define features (X) and target variable (y)
feature_cols = ['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_sec','interval_afstand', 'ervaring', 'rust_sec']
X = df[feature_cols]
y = df[['two_k_tijd_sec']]

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Predict
y_val_pred = clf.predict(X_val)
mse_test = mean_squared_error(y_val * 0.1, y_val_pred * 0.1)
print(mse_test)
print("Accuracy:",metrics.accuracy_score(y_val, y_val_pred))