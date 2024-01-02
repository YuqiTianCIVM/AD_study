'''
This script
1. creates a dictionary (all_data) where each key is a brain region.
2. For each region, it stores a dictionary with the region number repeated 30 times (or however many values there are) and corresponding neuron density values for each specimen.
3. After processing all files, it creates a DataFrame for each region and then concatenates these into a single DataFrame.
4. The first column of the final DataFrame lists the brain region numbers, and subsequent columns show neuron density values for each specimen.
5. Sort the specimen names and region names.
6. Add two empty rows per region to put the average and std.
'''


import os
import pandas as pd
import numpy as np
import re

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def process_folder(main_folder):
    all_data = {}  #Dictionary to hold data from all specimens

    #Loop through each subfolder (specimen) in the main folder
    for specimen_folder in sorted(os.listdir(main_folder), key=natural_sort_key):
        specimen_path = os.path.join(main_folder, specimen_folder)

        if os.path.isdir(specimen_path):
            #Loop through each file in the specimen subfolder
            for file in sorted(os.listdir(specimen_path), key=natural_sort_key):
                if file.endswith('_counts.csv'):
                    file_path = os.path.join(specimen_path, file)
                    df = pd.read_csv(file_path, header=None)  # Assuming CSV has no header

                    #Extracting brain region label from file name
                    brain_region = file.split('_')[0]
                    region_data = df.iloc[:, 0].tolist()  # Assuming density values are in the first column

                    if brain_region not in all_data:
                        all_data[brain_region] = {'Region': [brain_region]*len(region_data)}

                    all_data[brain_region][specimen_folder] = region_data

    #Create DataFrames for each region and insert empty rows
    region_dfs = []
    for region in sorted(all_data.keys(), key=natural_sort_key):
        data = all_data[region]
        df = pd.DataFrame(data)
        region_dfs.append(df)
        #Adding two empty rows
        empty_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
        region_dfs.append(empty_row)
        region_dfs.append(empty_row)

    combined_df = pd.concat(region_dfs, ignore_index=True)

    #Sort columns, keeping 'Region' as the first column
    sorted_cols = ['Region'] + sorted(combined_df.columns.drop('Region'), key=natural_sort_key)
    combined_df = combined_df[sorted_cols]

    return combined_df


# Usage
main_folder = r"S:\yt133\To_delete_Statistics\history\data_5xFAD_neuron_density".replace('\\','/')  # Replace with your folder path
combined_data = process_folder(main_folder)
combined_data.to_csv('combined_data.csv')  # Saving the combined data to a CSV file
