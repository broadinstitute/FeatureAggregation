"""
Import large .csv file(s) (>3 GB) and preprocess them so that they can be used during training/validation/testing
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import os
import pickle

data_type = 'AllTreatments'
plate_name = 'Plate00117013' #7013

# Load .csv
dirpath = r'/Users/rdijk/Documents/Data/ProcessedData'
file = f'{plate_name}_merged_{data_type}.csv'
filename = os.path.join(dirpath, file)
df = dd.read_csv(filename,
                 dtype={'broad_sample': str,
                        'gene': str,
                        'control_type': str})  # Specify offending columns' data type
# print(df.head())  # Top sample of .csv
# print(len(df))  # Total number of rows in .csv

# I fetch these from the .csv using DBeaver, have not found a method yet that does this efficiently here
filename2 = os.path.join(dirpath, f'Wells_{data_type}.csv')
wells = pd.read_csv(filename2)
#wells = pd.DataFrame(['E09', 'G17', 'H11', 'H21', 'K03', 'L08']) poscon_ORF
#wells = pd.DataFrame(['A02','A09','A17','E24','F01','M01','P05','P10']) # outer wells

output_dirName = f'datasets/unfiltered/DataLoader_{plate_name}_{data_type}'

try:
    os.mkdir(output_dirName)
except:
    pass

grouped_wells = df.groupby('well')
# print(grouped_wells.get_group('B03').compute()) # Gather all data from a specific well

for index, row in wells.iterrows():
    print(f"Index : {index}, Value: {row[0]}")
    cdf = grouped_wells.get_group(row[0]).compute()  # retrieve df of current well
    cell_features = cdf.iloc[:, 8:].to_numpy(dtype=float)
    well = pd.unique(cdf.iloc[:, 0])
    ctype = pd.unique(cdf.iloc[:, 3])
    gene = pd.unique(cdf.iloc[:, 4])

    assert well == row[0]
    assert len(well) == 1
    assert len(ctype) == 1
    assert len(gene) == 1

    dict = {
        'gene': gene[0],
        'control_type': ctype[0],
        'well_name': well[0],
        'cell_features': cell_features
    }
    print('Cell features array size: ', np.shape(cell_features))

    with open(os.path.join(output_dirName, f'{well[0]}.pkl'), 'wb') as f:
        pickle.dump(dict, f)

