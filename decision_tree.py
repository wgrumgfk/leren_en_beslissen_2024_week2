from sklearn import tree
import pandas as pd

df = pd.read_csv('okeanos_processed.csv')
df.dropna(how = 'any', subset=['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','two_k_watt','500_split_watt','aantal_intervallen','intervaltype','interval_afstand','interval_nummer','2k tijd','mean_watt_per_training'], inplace=True)

# Define features (X) and target variable (y)
X = df[['days_until_2k', 'zwaar','AT','I','ID','ED','500_split_watt','interval_afstand','mean_watt_per_training']]

# X = df[['days_until_2k', 'man', 'zwaar','AT','I','ID','ED','500_split_watt','interval_afstand','mean_watt_per_training']]
y = df['man']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
# clf.predict([[88, 0,1,1,1,0,0.0000000477,4000,0.0000000000765]])