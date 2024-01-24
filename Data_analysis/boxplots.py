import pandas as pd
import os
import matplotlib.pyplot as plt
import statistics
# Dataframe inladen voor Frédérique
csv_path = r"okeanos.csv"
df = pd.read_csv(csv_path)
# Convert time stamps to seconds

# Filter out rows with empty values
filtered_df = df.dropna(subset=['500_split', '2k tijd'])

def entry_to_seconds(entry):
    if pd.isna(entry):
        return None
    # Define your processing logic for each type of entry
    if ':' and '.' in entry:
        # Process entries like '1:47.4'
        minutes = pd.to_numeric(entry.split(':')[0])
        seconds = pd.to_numeric(entry.split(':')[1])
        return minutes * 60 + seconds
    if ':' and ',' in entry:
        # Process entries like '1:47,4'
        minutes = pd.to_numeric(entry.split(':')[0])
        seconds = pd.to_numeric(entry.split(':')[1].replace(',', '.'))
        return minutes * 60 + seconds
    else:
        # Process entries like '1.47.6'
        minutes = pd.to_numeric(entry.split('.')[0])
        seconds = pd.to_numeric(entry.split('.')[1])
        little_seconds = pd.to_numeric(entry.split('.')[2])
        return minutes * 60 + seconds + little_seconds/100

# Convert seconds to wattage
def seconds_to_wattage(filtered_df, column_name):
    # Apply the process_entry function to the specified column
    return 2.8 / filtered_df[column_name].apply(entry_to_seconds) ** 3

# Add 500 split wattage columns to dataframe
filtered_df.loc[:, '500_split_wattage']= (
    seconds_to_wattage(filtered_df, '500_split')
)

# Add 2k wattage columns to dataframe
filtered_df.loc[:, '2k_wattage']= (
    seconds_to_wattage(filtered_df, '2k tijd')
)

# Add rate difference column to dataframe
filtered_df.loc[:, 'rate_difference'] = (
    filtered_df['500_split_wattage'] / filtered_df['2k_wattage']
)

# Add difference column to dataframe
filtered_df.loc[:, 'difference'] = (
    filtered_df['500_split_wattage'] - filtered_df['2k_wattage']
)


def improvement_by_feature_and_person(filtered_df, feature):
    unique_values = filtered_df[feature].unique()
    unique_values = [x for x in unique_values if str(x) != 'nan']

    data = []

    for value in unique_values:
        men = filtered_df.loc[(df[feature] == value)]
        unique_men = men.naam.drop_duplicates().tolist()
        men_df = filtered_df[filtered_df['naam'].isin(unique_men)]
        men_improvement = men_df['rate_difference'].tolist()
        data.append(men_improvement)

    plt.boxplot(data, labels=unique_values, notch=None, vert=None, patch_artist=None, widths=None)
    plt.xlabel(feature)
    plt.ylabel('Wattage improvement 500m split training to 2k test')
    plt.title('Improvement by Zone')
    plt.show()

improvement_by_feature_and_person(filtered_df, 'zone')