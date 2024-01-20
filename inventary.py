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

# Completeness
raw_df = pd.read_csv(csv_path, delimiter=',', na_values=['', 'NA', 'N/A', 'NaN', 'nan'])
non_empty_df = raw_df.dropna(how='all', subset=["interval_afstand", "interval_tijd", "datum","ervaring","geslacht","gewichtsklasse","ploeg","naam","zone","trainingype","500_split","aantal_intervallen","intervaltype","interval_nummer","rust","machine","spm","2k tijd","2k datum"])
complete_df = raw_df.dropna(how='any', subset=["datum","ervaring","geslacht","gewichtsklasse","ploeg","naam","zone","trainingype","500_split","aantal_intervallen","intervaltype","interval_nummer","rust","machine","spm","2k tijd","2k datum"])

print(raw_df.shape)
print(non_empty_df.shape)
print(complete_df.shape)

print((1 - (complete_df.shape[0] / non_empty_df.shape[0])) * 100)

col_names = raw_df.columns.tolist()
seperated_df = dict()

for col in col_names:
    filled_entries = non_empty_df.dropna(how='any', subset=[col])  
    empty_entries = non_empty_df[~non_empty_df.index.isin(filled_entries.index)]
    seperated_df[col] = {'filled': filled_entries, 'empty': empty_entries}
# TO DO plot distribution of filled entries for every feature of the dataset.
plt.bar('other', len(seperated_df['datum']['filled'])/len(non_empty_df) * 100)



for col in col_names:
    # if len(seperated_df[col]['filled'])/len(non_empty_df) != 1.0:
    # All these columns are fully filled in in the dataset.
    if col not in ['datum', 'ervaring', 'geslacht', 'gewichtsklasse', 
                  'ploeg', 'naam', 'trainingype', 'aantal_intervallen', 
                  'intervaltype', 'interval_nummer', 'interval_afstand', 'interval_tijd']:
        plt.bar(str(col)[:11], len(seperated_df[col]['filled'])/len(non_empty_df) * 100)

plt.title("% filled row entries per column")
plt.xlabel("Column")
plt.yticks([10, 20, 30,40,50,60,70,80,90,100])
plt.xticks(fontsize = 6.5)
plt.ylim(0, 100)
plt.ylabel("% filled row entries")
plt.show()

print('Overig = datum, ervaring, geslacht, gewichtsklasse, ploeg, naam, trainingype, aantal_intervallen, intervaltype, interval_nummer' )
