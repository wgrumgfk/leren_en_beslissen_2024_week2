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
    return split_500_to_watt(float(split))

def average_split_per_person(dataframe):
    dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
    dataframe['average_speed'] = dataframe.groupby('naam')['tijd'].transform('mean')
    return dataframe

def average_split_per_person_and_zone(dataframe):
    dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
    dataframe['average_speed'] = dataframe.groupby(['naam', 'zone'])['tijd'].transform('mean')
    return dataframe

def average_split_per_person_and_type(dataframe):
    dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
    dataframe['average_speed'] = dataframe.groupby(['naam', 'intervaltype'])['tijd'].transform('mean')
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
men = df[(df['geslacht'] == 'M')]
women = df[(df['geslacht'] == 'V')]

average = average_split_per_person(df)
average = average.drop_duplicates(subset='naam', keep='first')

average_men = average[(average['geslacht'] == 'M')]
average_women = average[(average['geslacht'] == 'V')]

filtered_df = average[(average['500_split_sec'] > 95)]

average_men_filtered = average_men[(average_men['500_split_sec'] > 95)]

correlation = df['500_split_sec'].corr(df['two_k_tijd_sec'])
print(f'Correlation coefficient (under threshold): {correlation}')

# # Scatter plot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='500_split_sec', y='two_k_tijd_sec', hue='geslacht', data=average)
# plt.xlabel('Split training')
# plt.ylabel('Split 2k test')

# legend = plt.legend(title='Gender', title_fontsize=14, fontsize = 14)
# new_labels = ['M', 'F']
# for text, label in zip(legend.get_texts(), new_labels):
#     text.set_text(label)
# plt.grid(True)
# plt.show()

# # Scatter plot
# plt.figure(figsize=(10, 6))
# palette = {'LU': '#1f77b4', 'ZU': '#ff7f0e', 'LE': '#2ca02c', 'ZE': '#d62728'}
# sns.scatterplot(x='500_split_sec', y='two_k_tijd_sec', hue='group', data=average_women, palette = palette)
# # sns.regplot(x = "500_split_sec", y = "two_k_tijd_sec", data = men_filtered_df, scatter=False)
# plt.xlabel('Split training')
# plt.ylabel('Split 2k test')

# legend = plt.legend(title_fontsize=14, fontsize = 14, loc='lower right')
# new_labels = ['Light inexperienced', 'Heavy inexperienced', 'Light experienced', 'Heavy experienced']
# for text, label in zip(legend.get_texts(), new_labels):
#     text.set_text(label)
# plt.grid(True)
# plt.show()

# new_men_df = average_split_per_person_and_zone(men)
# new_men_df = new_men_df.drop_duplicates(subset=['naam', 'zone'], keep='first')

# new_women_df = average_split_per_person_and_zone(women)
# new_women_df = new_women_df.drop_duplicates(subset=['naam', 'zone'], keep='first')

# plt.figure(figsize=(10, 6))
# # palette = {'I': '#1f77b4', 'AT': '#ff7f0e', 'ED': '#2ca02c', 'ED+': '#d62728', 'ID': '#9467bd'}
# sns.scatterplot(x='500_split_sec', y='two_k_tijd_sec', hue='zone', data=new_women_df, palette='tab10')
# plt.xlabel('Split training')
# plt.ylabel('Split 2k test')

# legend = plt.legend(title_fontsize=14, fontsize = 14, loc='lower right')
# # new_labels = ['Light unexperienced', 'Heavy unexperienced', 'Light experienced', 'Heavy experienced']
# # for text, label in zip(legend.get_texts(), new_labels):
# #     text.set_text(label)
# plt.grid(True)
# plt.show()


# df['intervaltype'].str.strip()

# average_type = average_split_per_person_and_type(df)
# average_type = average.drop_duplicates(subset='naam', keep='first')

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='500_split_sec', y='two_k_tijd_sec', hue='intervaltype', data=average_type)
# plt.xlabel('Split training')
# plt.ylabel('Split 2k test')

# legend = plt.legend(title='Training type', title_fontsize=14, fontsize = 14)
# new_labels = ['Time', 'Distance']
# for text, label in zip(legend.get_texts(), new_labels):
#     text.set_text(label)
# plt.grid(True)
# plt.show()

# # x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# # y_axis = [112.493, 110.773, 117.12, 112.20, 119.50, 109.74, 114.45, 109.83, 114.95]

# # plt.bar(x_axis, y_axis)
# # plt.xlabel('Number of intervals')
# # plt.ylabel('Split training')
# # plt.ylim(100, 120)
# # plt.show()