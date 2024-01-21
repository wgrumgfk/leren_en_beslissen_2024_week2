import pandas as pd
import os
import matplotlib.pyplot as plt
from statistics import mean
from datetime import date


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

# Convert interval_tijd to interval_afstand
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
    return float(2.8 / (float((split/500) ** 3)))

# Calculate watt from 2k time in seconds.
def split_2k_to_watt(split):
    return split_500_to_watt(float(split / 4))

# Calculate 500 m split from input watt.
# If input is empty return ''   
def watt_to_pace(watt):
    if isinstance(watt, float):
        return float(500 * (float(2.8/float(watt))**(1/3)))
    
    return ''

# Load .csv file and return raw dataframe
def load_dataset():
    cwd = os.getcwd()
    csv_path = cwd + "\okeanos.csv"
    return pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])

# return absolute difference in days between 2 time notations of the form
# "dd-mm-yyyy'
# If the difference is 0 return ''.
def days_difference(date_training, date_2k):
    if (pd.isnull(date_2k)):
        return ''

    date_training_split = date_training.split('-')
    date_2k_split = date_2k.split('-')
    dtrain = date(int(date_training_split[2]), int(date_training_split[1]), int(date_training_split[0]))
    d2k = date(int(date_2k_split[2]), int(date_2k_split[1]), int(date_2k_split[0]))
    
    if abs((d2k - dtrain).days) > 0:
        return abs((d2k - dtrain).days)

# Return '' if input is 0 or empty value.
# Return seconds if input is in minutes.
def rust_seconden(rust):
    if not (isinstance(rust, str) or (rust == 0)):
        if rust < 10:
            return rust * 60
    
        return rust
    
    return ''


# input: dataframe and returns list of the same length
# of entries. Every entry is the mean watt or seconds corresponding to the
# current training for that particular 'name'.
def mean_500_per_training(input_df, watt_mode):

    cur_training_splits = []
    interval_count = 0
    output = []
    mean_500 = ""

    for index, row in input_df.iterrows():
        
        cur_interval_split = float(row['500_split_sec'])
        if watt_mode:
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
        if row['500_split_sec'] > 0:
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

    # Remove all completely empty rows
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

    #Add column with mean interval 500_split in watt for current training.
    col_mean_500_watt = mean_500_per_training(non_empty_df, True)
    non_empty_df.loc[:, 'mean_watt_per_training'] = col_mean_500_watt
    # non_empty_df['mean_watt_per_training'] = col_mean_500    # This column gives a SettingWithCopyWarning but is fully functional!
    #Add column with mean interval 500_split in seconds for current training.
    col_mean_500_secs = non_empty_df.apply(lambda x: watt_to_pace(x.mean_watt_per_training) , axis=1)
    non_empty_df.loc[:, 'mean_500_per_training'] = col_mean_500_secs
    # non_empty_df['mean_watt_per_training'] = col_mean_500    # This column gives a SettingWithCopyWarning but is fully functional!

    # Calculate distance for every interval and store as interval_afstand column.
    non_empty_df.loc[:, 'interval_afstand'] = non_empty_df.apply(time_to_distance, axis=1)
    # non_empty_df['interval_afstand'] = non_empty_df.apply(time_to_distance, axis=1)
    
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

    # Dummy categorial variables for man/vrouw
    col_man_dummy = non_empty_df.apply(lambda x: 1 if x.geslacht=='M' else 0 , axis=1)
    non_empty_df.insert(2, "man", col_man_dummy, True)
    col_vrouw_dummy = non_empty_df.apply(lambda x: 1 if x.geslacht=='V' else 0 , axis=1)
    non_empty_df.insert(3, "vrouw", col_vrouw_dummy, True)

    # Dummy categorial variables for zwaar/licht
    col_zwaar_dummy = non_empty_df.apply(lambda x: 1 if x.gewichtsklasse=='Z' else 0 , axis=1)
    non_empty_df.insert(4, "zwaar", col_zwaar_dummy, True)
    col_licht_dummy = non_empty_df.apply(lambda x: 1 if x.gewichtsklasse=='L' else 0 , axis=1)
    non_empty_df.insert(5, "licht", col_licht_dummy, True)

    # Dummy categorical variables voor zone 
    col_AT = non_empty_df.apply(lambda x: 1 if x.zone=='AT' else 0 , axis=1)
    non_empty_df.insert(6, "AT", col_AT, True)
    col_I = non_empty_df.apply(lambda x: 1 if x.zone=='I' else 0 , axis=1)
    non_empty_df.insert(7, "I", col_I, True)
    col_ID = non_empty_df.apply(lambda x: 1 if x.zone=='ID' else 0 , axis=1)
    non_empty_df.insert(8, "ID", col_ID, True)
    col_ED = non_empty_df.apply(lambda x: 1 if x.zone=='ED' else 0 , axis=1)
    non_empty_df.insert(9, "ED", col_ED, True)
    col_ED_plus = non_empty_df.apply(lambda x: 1 if x.zone=='ED+' else 0 , axis=1)
    non_empty_df.insert(10, "ED+", col_ED, True)

    #Dummy categorical variables voor trainingype
    # all types: 
    """
    ["5x5'", "3x15'", "3x20'", "3x2000m/5'r", '6000m', '4x1500m', "3x1000m/5'r",
    "6x500m/2'r", '1500m', "1'", '2000m', "30'", '3x2000m', "3x20'/3'r",
    "3x6x1'/1'r", 'minuutjes', '2x2000m', '6x500m', '3x1000m', "8x5'/3'r",
    "3x4000m/4'r", "4x8'/4'r", "3x10'", '1000m', '1500m + 500m', "4x8'", "8x3'",
    '4x750m', "3x10'/5'r", '2x2000m + 500m', "3x7x1'/1'r /3'r", '1000m + 500m',
    "3x5x1'/1'r /5'r", "4x5'", "3x12'", "2x25'", '8x500m', "4x2000/5'r", '100m',
    '500m', "3x20'/5'r", '1500m+500m', "HOP+3x1'r", '3x4000m', "3x7'", "3x8'",
    '1000m+500m', "3x10'/3'r", "3x2000/3'r", "4x5'/5'r", '2000', "3x12'/3'r",
    "4x500/5'r", "3x3000/5'r", "3x15'/3'r", "8x3'/3'r", "7x3'/3'r", "2x19'/3'r",
    "2x3000/5'r", "6x750/5'r", "2x2000/5'r", '1000', "3x8'/5'r", '1500m+750m',
    "3x12'/5'r", "2x6x1'/1'r", "2x7x1'/1'r", "3x11'", "6x6'", "3x13'", "5x8'", "20'",
    "3x2000m/4'r", "3x10'/4'r", "3x1000m/3'r", "8x3'/2'r", "6x750/3'r",
    "3x7x1'/1'r", "3x1500/5'r", "4x8'/5'r", '6000', "2x10'/5'r", "2x4x20''/40''r",
    "1x20'", "3x5x1'/1'r", "6x5'/2'r", "7x3'/2'r", "3x2000/5'r" ,"4x4x40''/40''r",
    "9x3'/2'r", "3x8x1'/1'r", "5x5'/3'r", "5x7'/3'r", "5x9'/3'r", "3x1000/3'r",
    "2x12'/8'r", "3x8x40'/20'r", "3x16'/3'r", "3x7'/5'r", "6x3'/3'r", "9x3'/3'r"]
    """
    all_trainingtypes = non_empty_df['trainingype'].unique()
    # print(all_trainingtypes)
    for trainingtype in all_trainingtypes:
        col_trainingtype = non_empty_df.apply(lambda x: 1 if x.trainingype==trainingtype else 0 , axis=1)
        non_empty_df.insert(6, str(trainingtype), col_trainingtype, True)

    # Add a rust_seconds column
    # Deze column verslechtert de presatie van het model helaas.
    col_rust_sec = non_empty_df.apply(lambda x: rust_seconden(x.rust) , axis=1)
    non_empty_df.insert(4, "rust_sec", col_rust_sec, True)

    # Add a dummy for intervaltype
    col_time = non_empty_df.apply(lambda x: 1 if x.intervaltype=='afstand' else 0 , axis=1)
    non_empty_df.insert(11, "afstand", col_time, True)

    # Delete unnecessary columns
    # trainingstype dummy variables maken?
    non_empty_df.drop(columns=['2k tijd', '500_split','rust', 'machine', 'two_k_datum','datum', 'geslacht', 'gewichtsklasse', 'ploeg', 'naam', 'intervaltype', 'trainingype', 'spm', 'zone'], inplace=True)

    print('exported processed dataframe with new columns to okeanos_processed.csv')


    non_empty_df.to_csv('okeanos_processed.csv')
