from sklearn import tree
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import tree

# Make sure all rows are complete
df = pd.read_csv('okeanos_processed.csv')
df.dropna(how = 'any', subset=['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','aantal_intervallen','interval_afstand','interval_nummer', 'two_k_tijd_sec'], inplace=True)


# Delete unecessary columns
df.dropna(how = 'any', subset=['two_k_tijd_sec', 'days_until_2k', 'man', 'zwaar','AT','I','ID','ED','ED+','aantal_intervallen','afstand','interval_afstand','interval_nummer'], inplace=True)
df['two_k_tijd_sec'] = df['two_k_tijd_sec'].multiply(10)
df['500_split_sec'] = df['two_k_tijd_sec'].multiply(10)
df.round({'two_k_tijd_sec': 1})
df['two_k_tijd_sec'] = df['two_k_tijd_sec'].astype(int)


# Define features (X) and target variable (y)
X = df[['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_sec','interval_afstand']]
feature_cols = ['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_sec','interval_afstand']
y = df[['two_k_tijd_sec']]

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Predict
y_val_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_val_pred))

# Visualize the decision tree for regression
# dot_data = export_graphviz(
#     regressor,
#     out_file=None,
#     feature_names=feature_cols,
#     filled=True,
#     rounded=True,
#     special_characters=True
# )

# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('decision_tree_regression.png')