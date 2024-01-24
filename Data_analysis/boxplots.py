import pandas as pd
import os
import matplotlib.pyplot as plt
import statistics
# Dataframe inladen voor Frédérique
csv_path = r"okeanos.csv"
df = pd.read_csv(csv_path)
# Convert time stamps to seconds

def split_500_to_watt(split):
    return float(2.8 / (float((split/500) ** 3)))

# Calculate watt from 2k time in seconds.
def split_2k_to_watt(split):
    return split_500_to_watt(float(split / 4))

# Filter out rows with empty values
non_empty_df = df.dropna(subset=['500_split', '2k tijd'])
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

df = non_empty_df
threshold = 400
filtered_df = df[(df['wattage'] < threshold) & (df['2k_wattage'] < threshold)]


# Add rate difference column to dataframe
filtered_df.loc[:, 'rate_difference'] = (
    filtered_df['wattage'] / filtered_df['2k_wattage']
)

# Add difference column to dataframe
filtered_df.loc[:, 'difference'] = (
    filtered_df['wattage'] - filtered_df['2k_wattage']
)

def improvement_by_feature_and_person(filtered_df, feature):
    unique_values = filtered_df[feature].unique()
    unique_values = [x for x in unique_values if str(x) != 'nan']

    data = []

    for value in unique_values:
        men = filtered_df.loc[(df[feature] == value)]
        unique_men = men.naam.drop_duplicates().tolist()
        men_df = filtered_df[filtered_df['naam'].isin(unique_men)]
        men_improvement = men_df['difference'].tolist()
        data.append(men_improvement)

    plt.boxplot(data, labels=unique_values, notch=None, vert=None, patch_artist=None, widths=None)
    plt.xlabel(feature)
    plt.ylabel('Wattage improvement 500m split training to 2k test')
    plt.title('Improvement by Zone')
    plt.show()

improvement_by_feature_and_person(filtered_df, 'zone')