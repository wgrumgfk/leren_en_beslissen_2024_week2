import pandas as pd
csv_path = r"okeanos_processed.csv"
df = pd.read_csv(csv_path)

feature = 'rust_sec'
correlation = df[feature].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient for {feature}:')
print(correlation)

feature = 'aantal_intervallen'
correlation = df[feature].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient for {feature}:')
print(correlation)

feature = 'calculated_distance'
correlation = df[feature].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient for {feature}:')
print(correlation)

feature = 'days_until_2k'
correlation = df[feature].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient for {feature}:')
print(correlation)

feature = '500_split_sec'
correlation = df[feature].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient for {feature}:')
print(correlation)

feature = '500_split_watt'
correlation = df[feature].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient for {feature}:')
print(correlation)

feature = 'mean_500_per_training'
correlation = df[feature].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient for {feature}:')
print(correlation)

feature = 'mean_watt_per_training'
correlation = df[feature].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient for {feature}:')
print(correlation)