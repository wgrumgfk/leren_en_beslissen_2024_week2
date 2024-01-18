from sklearn import tree
import pandas as pd

df = pd.read_csv('okeanos_processed_decision_tree.csv')
df.dropna(how = 'any', subset=['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','aantal_intervallen','intervaltype','interval_afstand','interval_nummer', 'two_k_tijd_sec'], inplace=True)

# Define features (X) and target variable (y)
X = df[['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_sec','interval_afstand']]
y = df[['two_k_tijd_sec']]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
predict = clf.predict([[88,0,0,0,0,1,0,390,4000]])
print(predict)