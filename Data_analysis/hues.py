import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns

def entry_to_seconds(entry):
    if pd.isna(entry):
        return None
    
    # Replace commas with dots
    entry = entry.replace(',', '.')
    
    # Split the entry based on both ':' and '.'
    split_chars = [':', '.']
    parts = None
    for char in split_chars:
        if char in entry:
            parts = entry.split(char)
            break
    
    if parts is None:
        raise ValueError("Invalid time string format")
    
    # Convert the parts to numeric values
    minutes = pd.to_numeric(parts[0])
    seconds = pd.to_numeric(parts[1])
    
    if len(parts) > 2:
        little_seconds = pd.to_numeric(parts[2])
        return minutes * 60 + seconds + little_seconds / 100
    else:
        return minutes * 60 + seconds
    

def time_notation_to_sec(time_notation):
    # Check if incoming time_notation is non-empty
    # time_notation 3 variations; 00:07:30.300 or 07:30.300 07.30.300
    if isinstance(time_notation, str):
        time_notation = time_notation.replace(',', '.')
        split_string = time_notation.split(':')
        if len(split_string) > 2:   # variant like 00:07:30.300
            sec = (float(split_string[1]) * 60) +  float(split_string[2])
        elif len(split_string) == 2:   # variant like 07:30.300
            sec = (float(split_string[0]) * 60) +  float(split_string[1])
        else:       # variant like 7:30.300
            sec = (float(time_notation[0] * 60) + float(time_notation[2:]))
    else:     # return empty string if empty entry
        return ''
    return sec
# Calculate watt from 500_m split in seconds.
def split_500_to_watt(split):
    return float(2.8 / (float((split/500) ** 3)))

# Calculate watt from 2k time in seconds.
def split_2k_to_watt(split):
    return split_500_to_watt(float(split / 4))

def average_split_per_person(dataframe):
    dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
    dataframe['average_speed'] = dataframe.groupby('naam')['tijd'].transform('mean')
    return dataframe

# load file
cwd = os.getcwd()
csv_path = cwd + "/okeanos.csv"
raw_df = pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])

# store column names in col_names
col_names = raw_df.columns.tolist()
print(raw_df.shape)
# Remove all completely empty rows or when theres only a single 2k date filled, 
non_empty_df = raw_df.dropna(how='all', subset=(col_names[:-1]))

col_500_split = non_empty_df.dropna(how='any', subset=('500_split')).loc[:,"500_split"]
col_500_split_sec = col_500_split.apply(time_notation_to_sec)
non_empty_df.insert(9, "500_split_sec", col_500_split_sec, True)

# Add watt column for 500m_split
col_500_split_sec = non_empty_df.dropna(how='any', subset=('500_split_sec')).loc[:,"500_split_sec"]
col_500_split_watt = col_500_split_sec.apply(split_500_to_watt)
non_empty_df.insert(10, "wattage", col_500_split_watt, True)

# Select all 2k_times entries and convert to seconds and insert new column into df
col_two_k = non_empty_df.dropna(how='any', subset=('2k tijd')).loc[:,"2k tijd"]
col_two_k_sec = col_two_k.apply(time_notation_to_sec)
non_empty_df.insert(1, "two_k_tijd_sec", col_two_k_sec, True)

# Add watt column for 2k time
col_two_k_tijd_sec = non_empty_df.dropna(how='any', subset=('two_k_tijd_sec')).loc[:,"two_k_tijd_sec"]
col_two_k_watt = col_two_k_tijd_sec.apply(split_2k_to_watt)
non_empty_df.insert(1, "2k_wattage", col_two_k_watt, True)

# Set the threshold
threshold = 400

df = non_empty_df
# only keep data below threshold
filtered_df = df[(df['wattage'] < threshold) & (df['2k_wattage'] < threshold)]

# only one entry per person with average
new_df = average_split_per_person(filtered_df)
new_df = new_df.drop_duplicates(subset='naam', keep='first')

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='zone', data=new_df)
sns.regplot(x = "wattage", y = "2k_wattage", data = new_df, scatter=False)
# sns.regplot(x = "wattage", y = "2k_wattage", data = new_df)
plt.title('Correlation between training wattage and 2k wattage (Man)')
plt.xlabel('training wattage')
plt.ylabel('2k wattage')

plt.legend(title='zone')
plt.grid(True)
plt.show()