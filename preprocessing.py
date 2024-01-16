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

# Calculate watt from 500_m split in seconds.
def split_to_watt(split):
    return (split / 500) / pow(0.25, 3)

# TODO
def days_until_2k(training_date, two_k_date):
    days = 0
    return days

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
    
if __name__ == "__main__":
    raw_df = load_dataset()
    # replace spaces with _ in columns
    #raw_df.columns = [col.replace(' ', '_') for col in raw_df.columns]
    raw_df = raw_df.rename(columns={"2k datum": "two_k_datum"})
    col_names = raw_df.columns.tolist()
    
    # Remove all completely empty rows or when theres only a single 2k date filled, 
    non_empty_df = raw_df.dropna(how='all', subset=(col_names[:-1]))

    #print(non_empty_df)

    # Add 500m split to seconds column 
    # Select all 500m_split entries and convert to seconds and insert new column into df
    col_500_split = non_empty_df.dropna(how='any', subset=('500_split')).loc[:,"500_split"]
    col_500_split_sec = col_500_split.apply(time_notation_to_sec)
    non_empty_df.insert(9, "500_split_sec", col_500_split_sec, True)
    
    # Add 2k time to seconds column 
    # Select all 2k_times entries and convert to seconds and insert new column into df
    col_two_k = non_empty_df.dropna(how='any', subset=('2k tijd')).loc[:,"2k tijd"]
    col_two_k_sec = col_two_k.apply(time_notation_to_sec)
    non_empty_df.insert(19, "2k_tijd_sec", col_two_k_sec, True)


    # Calculate amount days between training date and 2k date.
    #cols_datums = non_empty_df.dropna(how='any', subset=(['datum', '2k datum'])).loc[:,["500_split", "2k tijd"]]
    non_empty_df['days_until_2k'] = non_empty_df.apply(lambda x: days_difference(x.datum, x.two_k_datum), axis=1)
    
    #cols_datums_days = non_empty_df.apply(lambda x: f(x.datum, x.2k datum), axis=1)
    
    #non_empty_df.insert(19, "2k_tijd_sec", col_two_k_sec, True)
    #print(date_difference(two_k_split.loc[0],two_k_split.loc[10]))
    
    #print(non_empty_df)

    non_empty_df.to_csv('okeanos_processed.csv')