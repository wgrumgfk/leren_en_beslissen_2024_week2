import pandas as pd
import os
import matplotlib.pyplot as plt
from statistics import mean
from datetime import date

# FUNCTIONS ###########################################################################################

# convert Time notated in csv to amount of seconds.
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

def time_to_distance(row):
    if pd.isna(row['interval_afstand']):
        if row['interval_tijd'] == '6x60':
            return float(360) / (float(row['500_split_sec']) * 2)
        elif row['interval_tijd'] == '7x60/60r':
            return float(420) / (float(row['500_split_sec']) * 2)
        elif '5x60' in row['interval_tijd']:
            return float(300) / (float(row['500_split_sec']) * 2)
        elif row['interval_tijd'] == 'xx60' or '60/60':
            return None
        return float(row['interval_tijd']) / (float(row['500_split_sec']) * 2)
    else:
        return row['interval_afstand']

# Calculate watt from 500_m split in seconds.
def split_500_to_watt(split):
    return float(2.8 / (float(split) ** 3))

def split_2k_to_watt(split):
    return split_500_to_watt(float(split) / 4)

# Load .csv file and return raw dataframe
def load_dataset():
    cwd = os.getcwd()
    csv_path = cwd+"\okeanos.csv"
    return pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])

# return difference in days between 2 time notations of the form
# "dd-mm-yyyy'
def days_difference(date_training, date_2k):
    if (pd.isnull(date_2k)):
        return ''

    date_training_split = date_training.split('-')
    date_2k_split = date_2k.split('-')
    dtrain = date(int(date_training_split[2]), int(date_training_split[1]), int(date_training_split[0]))
    d2k = date(int(date_2k_split[2]), int(date_2k_split[1]), int(date_2k_split[0]))
    return (d2k - dtrain).days

# PRECPROCESSING ######################################################################################  
# input: dataframe and returns list with the same length
# of entries. Every entry is the mean corresponding to the
# current training for that particular 'name'.
def mean_500_per_training(input_df):

    cur_training_splits = []
    interval_count = 0
    output = []
    mean_500 = ""

    for index, row in input_df.iterrows():
        
        cur_interval_split = float(row['500_split_watt'])
        # Convert interval_nummer str to int.
        if row['interval_nummer'] == 'avg':
            interval_nr = 1
        else:
            interval_nr = int(row['interval_nummer'])

        # If new training encountered calculate mean of previous 
        # training and add to output list.
        if interval_nr == 1:
            # Only calculate mean if previous training had valid 500_split entries
            # otherwise mean = ''
            if (len(cur_training_splits)):
                mean_500 = sum(cur_training_splits) / len(cur_training_splits)

            # Add the mean the amount of times by the previous amount of training intervals.
            for i in range(0, interval_count):
                output.append(mean_500)

            # reset recorded splits and mean
            cur_training_splits = []
            mean_500 = ''

        # Add current split if a valid split time (not '')
        if row['500_split_watt'] > 0:
            cur_training_splits.append(cur_interval_split)
            interval_count = interval_nr
        else:
            interval_count = interval_nr


    # Add mean of very last training in dataset to column
    mean_500 = sum(cur_training_splits) / len(cur_training_splits)
    for i in range(0, interval_count):
        output.append(mean_500)

    return output


if __name__ == "__main__":

    raw_df = load_dataset()
    # replace spaces with _ in columns
    raw_df = raw_df.rename(columns={"2k datum": "two_k_datum"})
    col_names = raw_df.columns.tolist()

    # Remove all completely empty rows or when theres only a single 2k date filled,
    non_empty_df = raw_df.dropna(how='all', subset=(col_names[:-1]))

    # Add 500m split to seconds column
    # Select all 500m_split entries and convert to seconds and insert new column into df
    col_500_split = non_empty_df.dropna(how='any', subset=('500_split')).loc[:,"500_split"]
    col_500_split_sec = col_500_split.apply(time_notation_to_sec)
    non_empty_df.insert(9, "500_split_sec", col_500_split_sec, True)

    # Add watt column for 500m_split
    col_500_split_sec = non_empty_df.dropna(how='any', subset=('500_split_sec')).loc[:,"500_split_sec"]
    col_500_split_watt = col_500_split_sec.apply(split_500_to_watt)
    non_empty_df.insert(10, "500_split_watt", col_500_split_watt, True)

    #Add column with mean interval 500_split for current training.
    # interval_nr is 100 percent filled in!
    col_mean_500 = mean_500_per_training(non_empty_df)
    print(len(col_mean_500), len(non_empty_df))
    non_empty_df['mean_watt_per_training'] = col_mean_500    # This column gives a SettingWithCopyWarning but is fully functional!

    # Calculate distance for every interval
    non_empty_df['interval_afstand'] = non_empty_df.apply(time_to_distance, axis=1)
    
    # Add 2k time to seconds column 
    # Select all 2k_times entries and convert to seconds and insert new column into df
    col_two_k = non_empty_df.dropna(how='any', subset=('2k tijd')).loc[:,"2k tijd"]
    col_two_k_sec = col_two_k.apply(time_notation_to_sec)
    non_empty_df.insert(1, "two_k_tijd_sec", col_two_k_sec, True)

    # Add watt column for 2k time
    col_two_k_tijd_sec = non_empty_df.dropna(how='any', subset=('two_k_tijd_sec')).loc[:,"two_k_tijd_sec"]
    col_two_k_watt = col_two_k_tijd_sec.apply(split_2k_to_watt)
    non_empty_df.insert(1, "two_k_watt", col_two_k_watt, True)


    # Calculate amount days between training date and 2k date and add as days_until_2k column
    col_date_difference = non_empty_df.apply(lambda x: days_difference(x.datum, x.two_k_datum), axis=1)
    non_empty_df.insert(1, "days_until_2k", col_date_difference, True)

    # Add dummy 'geslacht' column
    # if geslacht == 0 then vrouw
    col_man_dummy = non_empty_df.apply(lambda x: 1 if x.geslacht=='M' else 0 , axis=1)
    non_empty_df.insert(2, "man", col_man_dummy, True)

    # Add dummy 'gewicht' column
    # if gewicht == 0 then licht
    col_zwaar_dummy = non_empty_df.apply(lambda x: 1 if x.gewichtsklasse=='Z' else 0 , axis=1)
    non_empty_df.insert(3, "zwaar", col_zwaar_dummy, True)

    # Add dummy 'AT' column
    # if AT == 0 then one of the others
    col_AT = non_empty_df.apply(lambda x: 1 if x.zone=='AT' else 0 , axis=1)
    non_empty_df.insert(4, "AT", col_AT, True)

    # Add dummy 'I' column
    # if AT == 0 then one of the others
    col_I = non_empty_df.apply(lambda x: 1 if x.zone=='I' else 0 , axis=1)
    non_empty_df.insert(5, "I", col_I, True)

    # Add dummy 'ID' column
    # if AT == 0 then one of the others
    col_ID = non_empty_df.apply(lambda x: 1 if x.zone=='ID' else 0 , axis=1)
    non_empty_df.insert(6, "ID", col_ID, True)

    # Add dummy 'ED' column
    # if ED == 0 then one of the others
    col_ED = non_empty_df.apply(lambda x: 1 if x.zone=='ED' else 0 , axis=1)
    non_empty_df.insert(7, "ED", col_ED, True)

    non_empty_df.drop(columns=['datum', 'geslacht', 'gewichtsklasse', 'ploeg', 'naam', 'trainingype', 'interval_tijd'])

    #TODO
    # Dummy catagorien voor ZONE

    #TODO
    #Dummy catagorien voor trainingstype

    #TODO
    # COLUMN voor trainingsafstand ongeacht of trainingstype tijd of interval is
    # Per interval    
    
    print('exported processed dataframe with new columns to okeanos_processed.csv')
    non_empty_df.to_csv('okeanos_processed.csv')
