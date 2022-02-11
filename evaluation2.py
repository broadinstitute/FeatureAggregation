## Standard libraries
import os
from tqdm import tqdm
import pandas as pd

# Seeds
import random
import numpy as np

## PyTorch
import torch
import torch.utils.data as data

# Custom libraries
from networks.SimpleMLPs import MLP
from dataloader_pickles import DataloaderTrainV4
from utils import CalculatePercentReplicating
import utils
import utils_benchmark
from pycytominer.operations.transform import RobustMAD


NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

# Set random seed for reproducibility
manualSeed = 42
# manualSeed = random.randint(1,10000) # use if you want new results
print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# %% Load model
save_name_extension = 'model_bestval_simpleMLP_V1'  # extension of the saved model
model_name = save_name_extension
print('Loading:', model_name)

BS = 32  # batch size
nr_cells = 400  # nr of cells sampled from each well (no more than 1200 found in compound plates)
input_dim = 1938 # 1938
kFilters = 1  # times DIVISION of filters in model
latent_dim = 256//kFilters
output_dim = 128//kFilters
model = MLP(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim, k=kFilters)

TrainSplit = 0.8
save_features_to_csv = False
path = r'wandb/run-20220210_192526-fv1jh0v8/files'
models = os.listdir(path)
fullpath = os.path.join(path, model_name)
if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()
# %% Load all data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/unfiltered'

# Set paths for all training/validation files
plateDirTrain1 = 'DataLoader_Plate00117010_unfiltered'
tdir1 = os.path.join(rootDir, plateDirTrain1)
plateDirTrain2 = 'DataLoader_Plate00117011_unfiltered'
tdir2 = os.path.join(rootDir, plateDirTrain2)
plateDirVal1 = 'DataLoader_Plate00117012_unfiltered'
vdir1 = os.path.join(rootDir, plateDirVal1)
plateDirVal2 = 'DataLoader_Plate00117013_unfiltered'
vdir2 = os.path.join(rootDir, plateDirVal2)

# Load csv for pair formation
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/JUMP_target_compound_metadata_wells.csv', index_col=False)

# Load all absolute paths to the pickle files for training
filenamesTrain1 = [os.path.join(tdir1, file) for file in os.listdir(tdir1)]
filenamesTrain2 = [os.path.join(tdir2, file) for file in os.listdir(tdir2)]
filenamesTrain1.sort()
filenamesTrain2.sort()
# and for validation
filenamesVal1 = [os.path.join(vdir1, file) for file in os.listdir(vdir1)]
filenamesVal1.sort()
filenamesVal2 = [os.path.join(vdir2, file) for file in os.listdir(vdir2)]
filenamesVal2.sort()

# Create preprocessing dataframe for both validation and training
metadata['plate1'] = filenamesTrain1
metadata['plate2'] = filenamesTrain2
metadata['plate3'] = filenamesVal1
metadata['plate4'] = filenamesVal2

# Filter the data and create numerical labels
df = utils.filterData(metadata, 'negcon', encode='pert_iname')
# Split the data into training and validation set
TrainTotal, ValTotal = utils.train_val_split(df, 0.8)

trainset = DataloaderTrainV4(TrainTotal, nr_cells=nr_cells)
valset = DataloaderTrainV4(ValTotal, nr_cells=nr_cells)

train_loader = data.DataLoader(trainset, batch_size=BS, shuffle=False, collate_fn=utils.my_collate,
                               drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)
val_loader = data.DataLoader(valset, batch_size=BS, shuffle=False, collate_fn=utils.my_collate,
                             drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

#%% Calculate NCEloss + Create feature dataframes
trainDF = pd.DataFrame()
valDF = pd.DataFrame()

print('Calculating Features')
with torch.no_grad():
    for idx, (points, labels) in enumerate(tqdm(train_loader)):
        points = points.to(device)
        feats, _ = model(points)
        # Append everything to dataframes
        c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
        trainDF = pd.concat([trainDF, c1])

# Validation
with torch.no_grad():
    for i, (points, labels) in enumerate(tqdm(val_loader)):
        points = points.to(device)
        feats, _ = model(points)

        # Append everything to dataframes
        c2 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
        valDF = pd.concat([valDF, c2])


#%% Rename columns and normalize features
# Rename feature columns
trainDF.columns = [f"f{x}" for x in range(trainDF.shape[1]-1)] + ['labels']
valDF.columns = [f"f{x}" for x in range(valDF.shape[1]-1)] + ['labels']
print('trainDF shape: ', trainDF.shape)
print('valDF shape:', valDF.shape)

# Robust MAD normalize features
scaler = RobustMAD()
fitted_scaler1 = scaler.fit(trainDF.iloc[:, :-1])
fitted_scaler2 = scaler.fit(valDF.iloc[:, :-1])

trainDF.iloc[:, :-1] = fitted_scaler1.transform(trainDF.iloc[:, :-1])
valDF.iloc[:, :-1] = fitted_scaler2.transform(valDF.iloc[:, :-1])

#%% Save all the dataframes to .csv files!
if save_features_to_csv:
    trainDF.to_csv(r'outputs/MLP_profiles_plate1.csv', index=False)
    valDF.to_csv(r'outputs/MLP_profiles_plate2.csv', index=False)

#%% Analyze feature distributions
#for df in [plate1df, plate2df, plate3df, plate4df]:
df = valDF.iloc[:, :-1] # Only pass the features
utils.featureCorrelation(df, 16)
utils.compoundCorrelation(df, 16)

utils.createUmap(valDF, 30) # need the labels for Umap

#%% Calculate Percent Replicating
print('Calculating Percent Replicating')

save_name = "TVsplit_generalized_MLP_noNegcon_pertIname_nR3"  # "TVsplit_allWells_gene_nR3"  ||  "TVsplit_OnlyControls_well_nR3"
group_by_feature = 'labels'

n_replicatesT = int(round(trainDF['labels'].value_counts().mean()))
n_replicatesV = int(round(valDF['labels'].value_counts().mean()))
print('Train nR, Val nR:', n_replicatesT,',', n_replicatesV)
n_samples = 10000

corr_replicating_df = pd.DataFrame()
for plates, nR in zip([trainDF, valDF], [n_replicatesT, n_replicatesV]):
    temp_df = CalculatePercentReplicating(plates, group_by_feature, nR, n_samples)
    corr_replicating_df = pd.concat([corr_replicating_df, temp_df], ignore_index=True)

print(corr_replicating_df[['Description', 'Percent_Replicating']].to_markdown(index=False))

utils_benchmark.distribution_plot(df=corr_replicating_df, output_file=f"{save_name}_PR.png", metric="Percent Replicating")

corr_replicating_df['Percent_Replicating'] = corr_replicating_df['Percent_Replicating'].astype(float)

plot_corr_replicating_df = (
    corr_replicating_df.rename(columns={'Modality':'Perturbation'})
    .drop(columns=['Null_Replicating','Value_95','Replicating'])
)
