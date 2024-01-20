# GET STARTED ################################################################################

# import packages
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns

print('aanpassing voor deze fime')

cwd = os.getcwd()
csv_path = cwd + "/okeanos.csv"
raw_df = pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])

# store column names in col_names
col_names = raw_df.columns.tolist()
print(raw_df.shape)
# Remove all completely empty rows or when theres only a single 2k date filled, 
non_empty_df = raw_df.dropna(how='all', subset=(col_names[:-1]))
print(len(non_empty_df) / len(raw_df), '% of raw datafile is non empty rows')

# store every colum as key in dictionary
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

# Specify here what dataframe you want to use 
# df = seperated_df[..]['filled'] merged with ..........
df = non_empty_df

# GENERAL FUNCTIONS ########################################################################

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


def average_split_per_person(dataframe):
    dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
    dataframe['average_speed'] = dataframe.groupby('naam')['tijd'].transform('mean')
    return dataframe

def average_split_per_person_and_zone(dataframe):
    dataframe['tijd'] = df['500_split'].apply(entry_to_seconds)
    dataframe['average_speed'] = dataframe.groupby(['naam', 'zone'])['tijd'].transform('mean')
    return dataframe

# FILLED ENTRIES PER COLUMN ################################################################

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
# plt.show()

print('Overig = datum, ervaring, geslacht, gewichtsklasse, ploeg, naam, trainingype, aantal_intervallen, intervaltype, interval_nummer' )

# Split the dataset in groups based on gender, experience and weight

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


# BARPLOT ####################################################################################

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
# plt.show()


# PIECHART ##########################################################################################

pie_chart_df =  pd.DataFrame(amt_row_per_group, index=x_labels)
# make the plot
pie_chart_df.plot(kind='pie', subplots=True, figsize=(8, 8), autopct='%1.0f%%')

# show the plot
plt.title('Percentage of data entries per group')
# plt.show()


# BOXPLOT 2K tijd #####################################################################################
i = 0
x_labels = ["EJML", "OJMZ", "EJMZ", "OJVL", "EJVL", "OJVZ", "EJVZ"]
rower_2k_per_group = {"EJML": dict(), "OJMZ": dict(), "EJMZ": dict(),
                      "OJVL": dict(), "EJVL": dict(), "OJVZ": dict(), "EJVZ": dict()}


# Ik gooi de datum van de 2k hier weg. ALs er meerdere 2k tijden voor
# een roeier zijn wil je de datum eigenlijk wel weten.

# Make a list of all the 2K times of unique rowers
for group in [EJML, OJMZ, EJMZ, OJVL, EJVL, OJVZ, EJVZ]:
    for rower in group.naam.drop_duplicates().tolist(): 
        #print(rower)
        time_set = set(group.loc[group['naam'] == rower]['2k tijd'].tolist())
        float_times = []
        for time in time_set:
            if isinstance(time, str):
                float_time = entry_to_seconds(time)
                float_times.append(float_time)
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
# plt.show()

# SCATTERPLOTS SORTED BY GENDER, EXPERIENCE AND WEIGHT #############################################

# calculate the wattage
df.loc[:, 'wattage']= (
    seconds_to_wattage(df, '500_split')
)

# divide the 2k time so that it becomes a 500-split
df['500_split_2k']= df['2k tijd'].apply(entry_to_seconds) / 4


df.loc[:, '2k_wattage'] = (
    2.8 / df['500_split_2k'] ** 3
)

# Set the threshold
threshold = 0.001

# only keep data below threshold
filtered_df = df[(df['wattage'] < threshold) & (df['2k_wattage'] < threshold)]

# only one entry per person with average
new_df = average_split_per_person(filtered_df)
new_df = new_df.drop_duplicates(subset='naam', keep='first')

# only one entry per man with average
men_df = df[df['geslacht'] == 'M']
men_df = average_split_per_person(men_df)
men_df = men_df.drop_duplicates(subset='naam', keep='first')

# only one entry per woman with average
women_df = new_df[new_df['geslacht'] == 'V']
women_df = average_split_per_person(women_df)
women_df = women_df.drop_duplicates(subset='naam', keep='first')

# Scatter plot
plt.figure(figsize=(10, 6))
sns.regplot(x = "wattage", y = "2k_wattage", data = filtered_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Under Threshold)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')

plt.legend()
plt.grid(True)
plt.show()

# Correlation information for the filtered data
correlation = filtered_df['wattage'].corr(filtered_df['2k_wattage'])
print(f'Correlation coefficient (under threshold): {correlation}')

# Scatter plot for men
plt.figure(figsize=(10, 6))
sns.regplot(x = "wattage", y = "2k_wattage", data = men_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Men)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.grid(True)
# plt.show()

# Correlation information for men
men_correlation = men_df['wattage'].corr(men_df['2k_wattage'])
print(f'Correlation coefficient for men: {men_correlation}')


# Scatter plot for women
plt.figure(figsize=(10, 6))
sns.regplot(x = "wattage", y = "2k_wattage", data = women_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Women)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.grid(True)
# plt.show()

# Correlation information for women with threshold
women_correlation = women_df['wattage'].corr(women_df['2k_wattage'])
print(f'Correlation coefficient for women (above threshold): {women_correlation}')

# Set the threshold
threshold = 0.001

# Scatter plot for men
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='ervaring', data=men_df)
sns.regplot(x = "wattage", y = "2k_wattage", data = men_df, scatter=False)
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

# Scatter plot for women
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='ervaring', data=women_df)
sns.regplot(x = "wattage", y = "2k_wattage", data = women_df, scatter=False)
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

# Set the threshold
threshold = 0.001

# Scatter plot for men
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='gewichtsklasse', data=men_df)
sns.regplot(x = "wattage", y = "2k_wattage", data = men_df, scatter=False)
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

# Scatter plot for women
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='gewichtsklasse', data=women_df)
sns.regplot(x = "wattage", y = "2k_wattage", data = women_df, scatter=False)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Women)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.legend(title='gewichtsklasse')
plt.grid(True)
# plt.show()

# Correlation information for women
women_correlation = women_df.groupby('gewichtsklasse')['wattage'].corr(women_df['2k_wattage'])
print(f'Correlation coefficient for women:')
print(women_correlation)

# SCATTERPLOTS SORTED BY ZONE #####################################################################

# TO DO: FIND A SYSTEM TO NOT HAVE DOUBLES
men_df = filtered_df[filtered_df['geslacht'] == 'M']
women_df = filtered_df[filtered_df['geslacht'] == 'V']

men_df = average_split_per_person_and_zone(men_df)
men_df = men_df.drop_duplicates(subset=['naam', 'zone'], keep='first')

women_df = average_split_per_person_and_zone(women_df)
women_df = women_df.drop_duplicates(subset=['naam', 'zone'], keep='first')


# Set the threshold
threshold = 0.001

# Scatter plot for men
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='zone', data=men_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Men)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.legend(title='zone')
plt.grid(True)
plt.show()

# Correlation information for men
men_correlation = men_df.groupby('zone')['wattage'].corr(men_df['2k_wattage'])
print(f'Correlation coefficient for men:')
print(men_correlation)

# Scatter plot for women
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wattage', y='2k_wattage', hue='zone', data=women_df)
plt.title('Correlation between 500m Split Wattage and 2k Wattage (Women)')
plt.xlabel('500m Split Wattage')
plt.ylabel('2k Wattage')
plt.legend(title='zone')
plt.grid(True)
plt.show()

# Correlation information for women
women_correlation = women_df.groupby('zone')['wattage'].corr(women_df['2k_wattage'])
print(f'Correlation coefficient for women:')
print(women_correlation)

# BOXPLOTS IMPROVENT ############################################################################

filtered_df = df.dropna(subset=['500_split', '2k tijd'])

filtered_df.loc[:, 'rate_difference'] = (
    filtered_df['wattage'] / filtered_df['2k_wattage']
)

print(filtered_df['rate_difference'])

filtered_df.loc[:, 'difference'] = (
    filtered_df['wattage'] - filtered_df['2k_wattage']
)

def improvement_by_feature_and_person(filtered_df, feature):
    filtered_df = filtered_df.dropna(subset=[feature])
    unique_values = filtered_df[feature].unique()
    print(unique_values)
    data = []

    for value in unique_values:
        men = filtered_df.loc[filtered_df[feature] == value]
        unique_men = men.naam.drop_duplicates().tolist()
        men_df = filtered_df[filtered_df['naam'].isin(unique_men)]
        men_improvement = men_df['rate_difference'].tolist()
        data.append(men_improvement)

    plt.boxplot(data, labels=unique_values, notch=None, vert=None, patch_artist=None, widths=None)
    plt.show()

improvement_by_feature_and_person(filtered_df, 'zone')