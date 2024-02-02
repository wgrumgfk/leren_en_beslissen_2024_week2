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

def time2k_to_sec(time_notation):
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
    return sec / 4
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
col_two_k_sec = col_two_k.apply(time2k_to_sec)
non_empty_df.insert(1, "two_k_tijd_sec", col_two_k_sec, True)

# Add watt column for 2k time
col_two_k_tijd_sec = non_empty_df.dropna(how='any', subset=('two_k_tijd_sec')).loc[:,"two_k_tijd_sec"]
col_two_k_watt = col_two_k_tijd_sec.apply(split_2k_to_watt)
non_empty_df.insert(1, "2k_wattage", col_two_k_watt, True)

# Set the threshold
threshold = 400

df = non_empty_df

df['group'] = np.where((df['gewichtsklasse'] == 'Z') & (df['ervaring'] == 1), 'ZE',
                                 np.where((df['gewichtsklasse'] == 'L') & (df['ervaring'] == 0), 'LU',
                                          np.where((df['gewichtsklasse'] == 'Z') & (df['ervaring'] == 0), 'ZU', 'LE')))

# only one entry per person with average
df = average_split_per_person(df)
df = df.drop_duplicates(subset='naam', keep='first')

men_df = df[(df['geslacht'] == 'M')]
women_df = df[(df['geslacht'] == 'V')]

filtered_df = df[(df['500_split_sec'] > 95)]

men_filtered_df = men_df[(men_df['500_split_sec'] > 95)]

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='500_split_sec', y='two_k_tijd_sec', hue='geslacht', data=df)
# sns.regplot(x = "500_split_sec", y = "two_k_tijd_sec", data = filtered_df, scatter=False)
plt.xlabel('Split training')
plt.ylabel('Split 2k test')

legend = plt.legend(title='Gender', title_fontsize=14, fontsize = 14)
new_labels = ['M', 'F']
for text, label in zip(legend.get_texts(), new_labels):
    text.set_text(label)
plt.grid(True)
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='500_split_sec', y='two_k_tijd_sec', hue='group', data=women_df)
# sns.regplot(x = "500_split_sec", y = "two_k_tijd_sec", data = men_filtered_df, scatter=False)
plt.xlabel('Split training')
plt.ylabel('Split 2k test')

legend = plt.legend(title_fontsize=14, fontsize = 14, loc='lower right')
new_labels = ['Light unexperienced', 'Heavy unexperienced', 'Light experienced', 'Heavy experienced']
for text, label in zip(legend.get_texts(), new_labels):
    text.set_text(label)
plt.grid(True)
plt.show()