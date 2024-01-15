import pandas as pd
import os
import matplotlib.pyplot as plt
from statistics import mean


cwd = os.getcwd()
csv_path = "/home/bardha/Leren Beslissen/okeanos.csv"
raw_df = pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])
# store column names in col_names
col_names = raw_df.columns.tolist()

# Remove all completely empty rows or when theres only a single 2k date filled, 
non_empty_df = raw_df.dropna(how='all', subset=(col_names[:-1]))
print(len(non_empty_df) / len(raw_df), '% of raw datafile is non empty rows')

# Store every colums as key in dictionary.
# Seperate missing/filled value entries in dataframe as value. 
seperated_df = dict()

for col in col_names:
    filled_entries = non_empty_df.dropna(how='any', subset=[col])  
    empty_entries = non_empty_df[~non_empty_df.index.isin(filled_entries.index)]
    seperated_df[col] = {'filled': filled_entries, 'empty': empty_entries}

""" Handig voor report om de data te beschrijven
report_analytics = seperated_df[KOLOM]['empty']
Kun je omschrijven of er specifieke afwijkingen zijn zoals 'machine' ineens niet meer ingevuld
"""



# TO DO plot distribution of filled entries for every feature of the dataset.
plt.bar('overig', len(seperated_df['datum']['filled'])/len(non_empty_df))

for col in col_names:
    # All these columns are fully filled in in the dataset.
    if col not in ['datum', 'ervaring', 'geslacht', 'gewichtsklasse', 
                  'ploeg', 'naam', 'trainingype', 'aantal_intervallen', 
                  'intervaltype', 'interval_nummer']:
        plt.bar(str(col)[:11], len(seperated_df[col]['filled'])/len(non_empty_df))

plt.title("% filled row entries per column")
plt.xlabel("Column")
plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(fontsize = 6.5)
plt.ylim(0.3, 1)
plt.ylabel("% filled row entries")
plt.show()

print('Overig = datum, ervaring, geslacht, gewichtsklasse, ploeg, naam, trainingype, aantal_intervallen, intervaltype, interval_nummer' )

# Specify here what dataframe you wanna use 
# df = seperated_df[..]['filled'] merged with ..........
df = non_empty_df

# OJML has 0 entries in the raw_data_file
#OJML = df.loc[(df['ervaring'] == 1) & (df['geslacht'] == "M") & (df['gewichtsklasse'] == "L")]
EJML = df.loc[(df['ervaring'] == 0) & (df['geslacht'] == "M") & (df['gewichtsklasse'] == "L")]
OJMZ = df.loc[(df['ervaring'] == 1) & (df['geslacht'] == "M") & (df['gewichtsklasse'] == "Z")]
EJMZ = df.loc[(df['ervaring'] == 0) & (df['geslacht'] == "M") & (df['gewichtsklasse'] == "Z")]
OJVL = df.loc[(df['ervaring'] == 1) & (df['geslacht'] == "V") & (df['gewichtsklasse'] == "L")]
# EJVL Has 1% entries in raw_data_file
EJVL = df.loc[(df['ervaring'] == 0) & (df['geslacht'] == "V") & (df['gewichtsklasse'] == "L")]
OJVZ = df.loc[(df['ervaring'] == 1) & (df['geslacht'] == "V") & (df['gewichtsklasse'] == "Z")]
EJVZ = df.loc[(df['ervaring'] == 0) & (df['geslacht'] == "V") & (df['gewichtsklasse'] == "Z")]


###  BARPLOT

i = 0
# Change the order and colours of the groups to make sense
x_labels = ["EJML", "OJMZ", "EJMZ", "OJVL", "EJVL", "OJVZ", "EJVZ"]
amt_row_per_group = []

for group in [EJML, OJMZ, EJMZ, OJVL, EJVL, OJVZ, EJVZ]:
    # Create bars
    amt_row_entries = amt_row_per_group.append(len(group))
    plt.bar(x_labels[i], len(group.naam.drop_duplicates()))
    i+= 1

plt.title("Amount of distinct rowers per group")
plt.xlabel("Group")
plt.ylabel("Amount of distinct rowers")
plt.show()


############################
# Piechart
pie_chart_df =  pd.DataFrame(amt_row_per_group, index=x_labels)
# make the plot
pie_chart_df.plot(kind='pie', subplots=True, figsize=(8, 8), autopct='%1.0f%%')

# show the plot
plt.title('Percentage of data entries per group')
plt.show()

i = 0
x_labels = ["EJML", "OJMZ", "EJMZ", "OJVL", "EJVL", "OJVZ", "EJVZ"]
rower_2k_per_group = {"EJML": dict(), "OJMZ": dict(), "EJMZ": dict(),
                      "OJVL": dict(), "EJVL": dict(), "OJVZ": dict(), "EJVZ": dict()}


# Ik gooi de datum van de 2k hier weg. ALs er meerdere 2k tijden voor
# een roeier zijn wil je de datum eigenlijk wel weten.
for group in [EJML, OJMZ, EJMZ, OJVL, EJVL, OJVZ, EJVZ]:
    for rower in group.naam.drop_duplicates().tolist(): 
        #print(rower)
        time_set = set(group.loc[group['naam'] == rower]['2k tijd'].tolist())
        float_times = []
        for time in time_set:
            if isinstance(time, str):
                split_string = time.split(':')
                # convert string to float according to 2k time notation
                if len(split_string) > 2:
                    str_to_float = (float(time.split(':')[1]) * 60) +  float((time.split(':')[2]))
                elif len(split_string) == 2:
                    str_to_float = (float(time.split(':')[0]) * 60) +  float((time.split(':')[1]))
                float_times.append(str_to_float)
                rower_2k_per_group[x_labels[i]][rower] = float_times         
            else:
                pass

    # Create bars
    #amt_row_entries = amt_row_per_group.append(len(group))
    #plt.bar(x_labels[i], group.
    i+= 1

for group, rower in rower_2k_per_group.items():
    all_2k_times = []
    for rower_time in (rower_2k_per_group[group].values()):
        for seperate_2k in rower_time:
            all_2k_times.append(seperate_2k)
    
    rower_2k_per_group[group]['all_2ks'] = all_2k_times

boxplot_2k_data = []
boxplot_xticks = []

for k, v in rower_2k_per_group.items():
    boxplot_2k_data.append(rower_2k_per_group[k]['all_2ks'])
    boxplot_xticks.append(k)

fig = plt.figure(figsize =(5, 5))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(boxplot_2k_data)
plt.xticks([1,2,3,4,5,6,7], boxplot_xticks)
 
# show plot
plt.xlabel("Group")
plt.ylabel("2k times")
plt.title("2k times per group")
plt.show()



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

def seconds_to_wattage(df, column_name):
    # Apply the process_entry function to the specified column
    return 2.8 / df[column_name].apply(entry_to_seconds) ** 3

df.loc[:, 'wattage']= (
    seconds_to_wattage(df, '500_split')
)
df.loc[:, '2k_wattage'] = (
    seconds_to_wattage(df, '2k tijd')
)
# Display column names
print(df.columns)

# Display the first few rows of the dataframe
print(df.head())


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the threshold
threshold = 0.001

# Filter the DataFrame
filtered_df = df[(df['wattage'] < threshold) & (df['2k_wattage'] < threshold)]

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', data=filtered_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Under Threshold)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')

# Add the y=x line
# plt.plot([0, 0.001], [0, 0.001], color='red', linestyle='--', linewidth=2, label='y=x')

plt.legend()
plt.grid(True)
plt.show()

# Correlation information for the filtered data
correlation = filtered_df['wattage'].corr(filtered_df['2k_wattage'])
print(f'Correlation coefficient (under threshold): {correlation}')

import seaborn as sns

# Filter the DataFrame for men
men_df = df[df['geslacht'] == 'M']

# Scatter plot for men
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', data=men_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Men)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.grid(True)
plt.show()

# Correlation information for men
men_correlation = men_df['wattage'].corr(men_df['2k_wattage'])
print(f'Correlation coefficient for men: {men_correlation}')

# Filter the DataFrame for women
# Filter the DataFrame for women
women_df = df[df['geslacht'] == 'V']

# Set the threshold
threshold = 0.001

# Filter the DataFrame based on the threshold
filtered_women_df = women_df[(women_df['wattage'] < threshold) & (women_df['2k_wattage'] < threshold)]

# Scatter plot for women
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', data=filtered_women_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Women)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.grid(True)
plt.show()

# Correlation information for women with threshold
women_correlation = filtered_women_df['wattage'].corr(filtered_women_df['2k_wattage'])
print(f'Correlation coefficient for women (above threshold): {women_correlation}')




import seaborn as sns

# Set the threshold
threshold = 0.001

# Filter the DataFrame for men
men_df = df[(df['geslacht'] == 'M') & (df['wattage'] > threshold) & (df['2k_wattage'] > threshold)]

# Scatter plot for men
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='ervaring', data=men_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Men)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.legend(title='Experience')
plt.grid(True)
plt.show()

# Correlation information for men
men_correlation = men_df.groupby('ervaring')['wattage'].corr(men_df['2k_wattage'])
print(f'Correlation coefficient for men:')
print(men_correlation)

# Filter the DataFrame for women
women_df = df[(df['geslacht'] == 'V') & (df['wattage'] >  threshold) & (df['2k_wattage'] > threshold)]

# Scatter plot for women
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='ervaring', data=women_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Women)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.legend(title='Experience')
plt.grid(True)
plt.show()

# Correlation information for women
women_correlation = women_df.groupby('ervaring')['wattage'].corr(women_df['2k_wattage'])
print(f'Correlation coefficient for women:')
print(women_correlation)






import seaborn as sns

# Set the threshold
threshold = 0.001

# Filter the DataFrame for men
men_df = df[(df['geslacht'] == 'M') & (df['wattage'] < threshold) & (df['2k_wattage'] < threshold)]

# Scatter plot for men
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='gewichtsklasse', data=men_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Men)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.legend(title='Gewichtsklasse')
plt.grid(True)
plt.show()

# Correlation information for men
men_correlation = men_df.groupby('gewichtsklasse')['wattage'].corr(men_df['2k_wattage'])
print(f'Correlation coefficient for men:')
print(men_correlation)

# Filter the DataFrame for women
women_df = df[(df['geslacht'] == 'V') & (df['wattage'] <  threshold) & (df['2k_wattage'] < threshold)]

# Scatter plot for women
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='gewichtsklasse', data=women_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Women)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.legend(title='gewichtsklasse')
plt.grid(True)
plt.show()

# Correlation information for women
women_correlation = women_df.groupby('gewichtsklasse')['wattage'].corr(women_df['2k_wattage'])
print(f'Correlation coefficient for women:')
print(women_correlation)



