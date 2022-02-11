
# info_nce_loss
import torch
import torch.nn.functional as F
import datetime

# percent replicating
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import utils_benchmark

# heatmapCorrelation
import seaborn as sns
# Umap
import umap
import umap.plot
import colorcet as cc


def now():
    return str(datetime.datetime.now())+': '

def my_collate(batch):
    data = [item[0] for item in batch]
    data = torch.cat(data, dim=0)
    target = [item[1] for item in batch]
    target = torch.cat(target, dim=0)
    return [data, target]

def train_val_split(metadata_df, Tsplit=0.8):
    df = metadata_df[['labels', 'plate1']]
    df = pd.concat([df.rename(columns={'plate1': 'plate2'}), metadata_df[['labels', 'plate2']]])
    df = pd.concat([df.rename(columns={'plate2': 'plate3'}), metadata_df[['labels', 'plate3']]])
    df = pd.concat([df.rename(columns={'plate3': 'plate4'}), metadata_df[['labels', 'plate4']]])
    df = df.rename(columns={'plate4': 'well_path'})

    split = int(len(df.index)*Tsplit)

    return [df.iloc[:split, :].reset_index(drop=True), df.iloc[split:, :].reset_index(drop=True)]

def filterData(df, filter, encode=None, mode='default'):
    if 'negcon' in filter: # drop all negcon wells
        if mode == 'default':
            df = df[df.control_type != 'negcon']
        elif mode == 'eval':
            df = df[df.Metadata_control_type != 'negcon']
        df = df.reset_index(drop=True)
    if encode!=None:
        if mode=='default':
            pd.options.mode.chained_assignment = None  # default='warn'
            obj_df = df[encode].astype('category')
            df['labels'] = obj_df.cat.codes
            df = df.sort_values(by='labels')
            df = df.reset_index(drop=True)
        elif mode == 'eval':
            pd.options.mode.chained_assignment = None  # default='warn'
            obj_df = df[encode].astype('category')
            df['Metadata_labels'] = obj_df.cat.codes
            df = df.sort_values(by='Metadata_labels')
            df = df.reset_index(drop=True)
    return df





#######################
#%% EVALUATION STUFF ##
#######################
def createUmap(df, nSamples, mode='default'):
    plt.figure(figsize=(14, 10), dpi=300)
    labels = df.Metadata_labels.iloc[:nSamples]
    if mode == 'default':
        features = df.iloc[:nSamples, :-1]
    elif mode == 'old':
        features = df.iloc[:nSamples, 12:]
    reducer = umap.UMAP()
    embedding = reducer.fit(features)
    umap.plot.points(embedding, labels=labels, theme='fire')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP')
    plt.show()
    return


def compoundCorrelation(df, Ncompounds=20):
    plt.figure(figsize=(14, 10), dpi=300)
    df = df.transpose()
    df = df.iloc[:, :Ncompounds]
    Var_Corr = df.corr()
    #plot the heatmap
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    plt.title('compound Correlation')
    plt.show()
    return


def featureCorrelation(df, Nfeatures=20):
    plt.figure(figsize=(14, 10), dpi=300)
    df = df.iloc[:, :Nfeatures]
    Var_Corr = df.corr()
    #plot the heatmap
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    plt.title('feature Correlation')
    plt.show()
    return

def CalculatePercentReplicating(dfs, group_by_feature, n_replicates, n_samples=10000, description='Unknown'):
    """

    :param dfs: list of plate dataframes that are analysed together.
    :param group_by_feature: feature column which is used to make replicates
    :param n_replicates: number of expected replicates present in the given dataframes 'dfs' based on the 'group_by_feature' column
    :param n_samples: number of samples used to calculate the null distribution, often 10000
    :return: dataframe consisting of the calculated metrics
    """
    # Set plotting and sampling parameters
    random.seed(9000)
    plt.style.use("seaborn-ticks")
    plt.rcParams["image.cmap"] = "Set1"
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

    corr_replicating_df = pd.DataFrame()

    # Merge dfs in list
    try:
        data_df = pd.concat(dfs)
    except:
        data_df = dfs
    print('Created df of size: ', data_df.shape)
    metadata_df = utils_benchmark.get_metadata(data_df)
    features_df = utils_benchmark.get_featuredata(data_df).replace(np.inf, np.nan).dropna(axis=1, how="any")
    data_df = pd.concat([metadata_df, features_df], axis=1)

    replicating_corr = list(utils_benchmark.corr_between_replicates(data_df, group_by_feature))  # signal distribution
    null_replicating = list(utils_benchmark.corr_between_non_replicates(data_df, n_samples=n_samples, n_replicates=n_replicates,
                                                              metadata_compound_name=group_by_feature))  # null distribution

    prop_95_replicating, value_95_replicating = utils_benchmark.percent_score(null_replicating,
                                                                    replicating_corr,
                                                                    how='right')


    corr_replicating_df = corr_replicating_df.append({'Description': f'{description}',
                                                      'Replicating': replicating_corr,
                                                      'Null_Replicating': null_replicating,
                                                      'Percent_Replicating': '%.1f' % prop_95_replicating,
                                                      'Value_95': value_95_replicating}, ignore_index=True)

    return corr_replicating_df

