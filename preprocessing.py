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

def time_entry_to_distance(time_string):
    return None


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

# input: dataframe and returns pd.Series object of same length
# with mean for every training
# interval_nummer = avg or 1-9
# het komt voor dat maar een deel van de intervallen per training is ingevuld
def mean_500_per_training(df):
    cur_training_splits = []
    interval_count = 0
    column_entry_list = []
    mean_500 = 0

    for index, row in df.iterrows():
        if ['interval_nummer'] in [1, "avg"]:
            mean_500 = sum(cur_training_splits) / len(cur_training_splits)
            for i in range(0, interval_count):
                column_entry_list.append(mean_500)

        elif row['500_split_watt'] > 0:
            cur_training_splits.append(float(row['500_split_watt']))
            interval_count = row['interval_nummer']
        else:
            print(index, 'index, is < 0, dit is de entry: ', row['500_split_watt'] )
            pass


    print(column_entry_list)
    return 

    
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



    #TODO 
    #Add column with mean interval 500_split for current training.
    # interval_nr is 100 percent filled in!
    col_mean_500 = mean_500_per_training(non_empty_df)
    #non_empty_df.insert(11, "mean_500_current_training", col_mean_500, True)




    # Add 2k time to seconds column 
    # Select all 2k_times entries and convert to seconds and insert new column into df
    col_two_k = non_empty_df.dropna(how='any', subset=('2k tijd')).loc[:,"2k tijd"]
    col_two_k_sec = col_two_k.apply(time_notation_to_sec)
    non_empty_df.insert(20, "two_k_tijd_sec", col_two_k_sec, True)

    # Add watt column for 2k time
    col_two_k_tijd_sec = non_empty_df.dropna(how='any', subset=('two_k_tijd_sec')).loc[:,"two_k_tijd_sec"]
    col_two_k_watt = col_two_k_tijd_sec.apply(split_2k_to_watt)
    non_empty_df.insert(21, "two_k_watt", col_two_k_watt, True)

    # Calculate amount days between training date and 2k date and add as days_until_2k column
    col_date_difference = non_empty_df.apply(lambda x: days_difference(x.datum, x.two_k_datum), axis=1)
    non_empty_df.insert(23, "days_until_2k", col_date_difference, True)

    # Add dummy Man column
    # if man = 0 then vrouw
    col_man_dummy = non_empty_df.apply(lambda x: 1 if x.geslacht=='M' else 0 , axis=1)
    non_empty_df.insert(3, "Man_dummy", col_man_dummy, True)

    # Add dummy zwaar column
    # if zwaar = 0 then licht
    col_zwaar_dummy = non_empty_df.apply(lambda x: 1 if x.gewichtsklasse=='Z' else 0 , axis=1)
    non_empty_df.insert(5, "Zwaar_dummy", col_zwaar_dummy, True)

    #TODO
    # Dummy catagorien voor ZONE

    #TODO
    #Dummy catagorien voor trainingstype

    #TODO
    # COLUMN voor trainingsafstand ongeacht of trainingstype tijd of interval is
    # Per interval




    print('exported processed dataframe with new columns to okeanos_processed.csv')
    non_empty_df.to_csv('okeanos_processed.csv')