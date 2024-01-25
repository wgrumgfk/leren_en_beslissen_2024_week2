import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns

# # Convert interval_tijd to interval_afstand
# def time_to_distance(row):
#     if pd.notna(row['interval_tijd']) and pd.notna(row['500_split_sec']):
#         if row['interval_tijd'] == '6x60':
#             return float(360) / (float(row['500_split_sec']) * 2)
#         elif row['interval_tijd'] == '7x60/60r' or row['interval_tijd'] == '7x60':
#             return float(420) / (float(row['500_split_sec']) * 2)
#         elif '5x60' in row['interval_tijd']:
#             return float(300) / (float(row['500_split_sec']) * 2)
#         elif row['interval_tijd'] == '4x20':
#             return float(80) / (float(row['500_split_sec']) * 2)
#         elif row['interval_tijd'] == '7x1':
#             return float(420) / (float(row['500_split_sec']) * 2)
#         elif row['interval_tijd'] == '4x40':
#             return float(160) / (float(row['500_split_sec']) * 2)
#         elif row['interval_tijd'] == '8x60':
#             return float(480) / (float(row['500_split_sec']) * 2)
#         elif row['interval_tijd'] == 'xx60' or row['interval_tijd'] == '60/60'  or row['interval_tijd'] == 'xx40'  or row['interval_tijd'] == 'xx480':
#             return None
#         else:
#             #return float(row['interval_tijd']) / (float(row['500_split_sec']) * 2)    hoezo 2??
#             return (float(row['interval_tijd']) / (float(row['500_split_sec'])) * 500)
#     else:
#         return row['interval_afstand']
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
    

# def time_notation_to_sec(time_notation):
#     # Check if incoming time_notation is non-empty
#     # time_notation 3 variations; 00:07:30.300 or 07:30.300 07.30.300
#     if isinstance(time_notation, str):
#         time_notation = time_notation.replace(',', '.')
#         split_string = time_notation.split(':')
#         if len(split_string) > 2:   # variant like 00:07:30.300
#             sec = (float(split_string[1]) * 60) +  float(split_string[2])
#         elif len(split_string) == 2:   # variant like 07:30.300
#             sec = (float(split_string[0]) * 60) +  float(split_string[1])
#         else:       # variant like 7:30.300
#             sec = (float(time_notation[0] * 60) + float(time_notation[2:]))
#     else:     # return empty string if empty entry
#         return ''
#     return sec
# # Calculate watt from 500_m split in seconds.
# def split_500_to_watt(split):
#     return float(2.8 / (float((split/500) ** 3)))

# # Calculate watt from 2k time in seconds.
# def split_2k_to_watt(split):
#     return split_500_to_watt(float(split / 4))

# def average_split_per_person(dataframe):
#     dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
#     dataframe['average_speed'] = dataframe.groupby('naam')['tijd'].transform('mean')
#     return dataframe

# # load file
# cwd = os.getcwd()
# csv_path = cwd + "/okeanos.csv"
# raw_df = pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])

# # store column names in col_names
# col_names = raw_df.columns.tolist()
# print(raw_df.shape)
# # Remove all completely empty rows or when theres only a single 2k date filled, 
# non_empty_df = raw_df.dropna(how='all', subset=(col_names[:-1]))

# col_500_split = non_empty_df.dropna(how='any', subset=('500_split')).loc[:,"500_split"]
# col_500_split_sec = col_500_split.apply(time_notation_to_sec)
# non_empty_df.insert(9, "500_split_sec", col_500_split_sec, True)

# # Add watt column for 500m_split
# col_500_split_sec = non_empty_df.dropna(how='any', subset=('500_split_sec')).loc[:,"500_split_sec"]
# col_500_split_watt = col_500_split_sec.apply(split_500_to_watt)
# non_empty_df.insert(10, "wattage", col_500_split_watt, True)

# # Select all 2k_times entries and convert to seconds and insert new column into df
# col_two_k = non_empty_df.dropna(how='any', subset=('2k tijd')).loc[:,"2k tijd"]
# col_two_k_sec = col_two_k.apply(time_notation_to_sec)
# non_empty_df.insert(1, "two_k_tijd_sec", col_two_k_sec, True)

# # Add watt column for 2k time
# col_two_k_tijd_sec = non_empty_df.dropna(how='any', subset=('two_k_tijd_sec')).loc[:,"two_k_tijd_sec"]
# col_two_k_watt = col_two_k_tijd_sec.apply(split_2k_to_watt)
# non_empty_df.insert(1, "2k_wattage", col_two_k_watt, True)

# # Calculate distance for every interval and store as interval_afstand column.
# # non_empty_df.loc[:, 'interval_afstand'] = non_empty_df['interval_afstand'].apply(time_to_distance)
# # non_empty_df['interval_afstand'] = non_empty_df.apply(time_to_distance, axis=1)

# # non_empty_df['calculated_distance'] = non_empty_df.apply(time_to_distance, axis=1)
# non_empty_df.loc[:, 'calculated_distance'] = non_empty_df.apply(time_to_distance, axis=1)

# # Replace NaN values in 'distance' with calculated values
# # non_empty_df['interval_afstand'] = non_empty_df['calculated_distance'].combine_first(non_empty_df['interval_afstand'])
# non_empty_df['interval_afstand'] = non_empty_df['calculated_distance'].combine_first(non_empty_df['interval_afstand'])


# # Set the threshold
# threshold = 400

# df = non_empty_df
# # only keep data below threshold
# filtered_df = df[(df['wattage'] < threshold) & (df['interval_afstand'] )]

# # only one entry per person with average
# #new_df = average_split_per_person(filtered_df)
# new_df = filtered_df.drop_duplicates(subset='naam', keep='first')
 
# print(new_df)
# # Scatter plot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='interval_afstand', y='wattage', data=new_df)
# sns.regplot(x = "interval_afstand", y = "wattage", data = new_df, scatter=False)
# # sns.regplot(x = "wattage", y = "2k_wattage", data = new_df)
# plt.title('Correlation between training wattage and 2k distance ')
# plt.xlabel('distance')
# plt.ylabel( 'training wattage')

# # plt.legend(title='zone')
# # plt.grid(True)
# plt.show()


# load file

csv_path ="/home/bardha/Documents/GitHub/leren_en_beslissen_2024_week2/okeanos_processed.csv"
raw_df = pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])

df = raw_df 
def average_split_per_person(dataframe):
    dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
    dataframe['average_speed'] = dataframe.groupby('naam')['tijd'].transform('mean')
    return dataframe


csv_path = "/home/bardha/Documents/GitHub/leren_en_beslissen_2024_week2/okeanos_processed.csv"
raw_df = pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])

df = raw_df 
def average_split_per_person(dataframe):
    dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
    dataframe['average_speed'] = dataframe.groupby('naam')['tijd'].transform('mean')
    return dataframe

plt.figure(figsize=(10, 6))
sns.scatterplot(x='calculated_distance', y='two_k_tijd_sec',  data=raw_df)
#sns.regplot(x="calculated_distance", y="two_k_tijd_sec", data=raw_df, scatter=False)
# sns.regplot(x="wattage", y="2k_wattage", data=new_df)
plt.title('Correlation between training wattage and 2k wattage per Rower')
plt.xlabel('afstand')
plt.ylabel('2k time')

# Set x-axis limits to 0 to 100


# Commenting out the legend as it was mentioned in the previous discussion
# plt.legend(title='zone')

plt.grid(True)
plt.show()
