import sys
import argparse

from datetime import datetime
import os
import scripts.utils_forTraining as utils
import pandas as pd
import numpy as np

from EPInformer.models import EPInformer_v2, enhancer_predictor_256bp
from scipy import stats
from tqdm import tqdm
import torch
from torch.utils.data import Subset, Dataset
import optuna


def generate_splits(df, group_name, n_folds=12, seed = 42):
    np.random.seed(seed)
    #df = df.reset_index()
    groups = df[group_name]
    chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr11', 'chr12', 'chr8',  'chr10', 'chr17', 'chr9', 'chr16', 'chr19', 'chr15', 'chrX', 'chr20', 'chr13', 'chr14', 'chr18', 'chr21', 'chr22'] 
    folds = []
    
    for i in range(0, n_folds):
        # test_groups
        if (23-i >= len(chrs)): 
          test_groups = [chrs[i]]
        else:
          test_groups = [chrs[i], chrs[23-i]]
        # valid_groups
        if i >= 11:
          val_groups = [chrs[0]]
        else:
          val_groups = [chrs[i+1], chrs[22-i]]
        print(test_groups)
        print(val_groups)
        
        test_idx = np.where(np.isin(groups, test_groups))[0]
        val_idx = np.where(np.isin(groups, val_groups))[0]
        train_idx = np.where(~np.isin(groups, test_groups + val_groups))[0]

        folds.append({
            'train_idx': df.loc[train_idx, 'Ensembl_ID'],
            'val_idx': df.loc[val_idx, 'Ensembl_ID'],
            'test_idx': df.loc[test_idx, 'Ensembl_ID']
        })

    print(f"Generated {len(folds)} round-robin paired folds.")
    return folds



def load_pretrained_splits(df, group_name, cv_file):

    groups = df[group_name]
    cv_df = pd.read_csv(cv_file, sep='\t', dtype={ 'chr': str })

    folds = []
    # detect how many folds from column names (assumes columns like fold1.test, fold1.valid, …)
    n_folds = sorted({ int(col.split('.')[0][4:])
                         for col in cv_df.columns
                         if col.startswith('fold') })

    for i in n_folds:
        test_mask  = cv_df[f'fold{i}.test']  == 1
        valid_mask = cv_df[f'fold{i}.valid'] == 1
        # train is everything not in test or valid
        train_mask = ~(test_mask | valid_mask)

        train_groups = cv_df.loc[train_mask , 'chr'].tolist() 
        val_groups = cv_df.loc[valid_mask, 'chr'].tolist() 
        test_groups = cv_df.loc[test_mask , 'chr'].tolist()

        test_idx = np.where(np.isin(groups, test_groups))[0]
        val_idx = np.where(np.isin(groups, val_groups))[0]
        train_idx = np.where(~np.isin(groups, test_groups + val_groups))[0]

        folds.append({
            'train_idx': df.loc[train_idx, 'Ensembl_ID'],
            'val_idx': df.loc[val_idx, 'Ensembl_ID'],
            'test_idx': df.loc[test_idx, 'Ensembl_ID']
        })

    print(f"Replicated {len(folds)} pretrained folds.")
    return folds




def generate_random_splits(df, group_name, n_folds=3, seed=42, verbose=True):
    np.random.seed(seed)
    
    groups = df[group_name].unique()
    chrs = sorted(groups)  # or use a custom list
    np.random.shuffle(chrs)

    # Split chromosomes into roughly equal folds
    chr_folds = np.array_split(chrs, n_folds)

    folds = []

    for i in range(n_folds):
        test_chrs = chr_folds[i]
        val_chrs = chr_folds[(i + 1) % n_folds]  # next fold for validation
        train_chrs = [chr for j, fold in enumerate(chr_folds) if j not in [i, (i + 1) % n_folds] for chr in fold]

        if verbose:
            print(f"Fold {i}:")
            print(f"  Test: {list(test_chrs)}")
            print(f"  Val : {list(val_chrs)}")
            print(f"  Train: {list(train_chrs)}\n")

        test_idx = df[df[group_name].isin(test_chrs)].index
        val_idx = df[df[group_name].isin(val_chrs)].index
        train_idx = df[df[group_name].isin(train_chrs)].index

        folds.append({
            'train_idx': df.loc[train_idx, 'Ensembl_ID'],
            'val_idx': df.loc[val_idx, 'Ensembl_ID'],
            'test_idx': df.loc[test_idx, 'Ensembl_ID']
        })

    print(f"✅ Generated {len(folds)} folds. Each chromosome appears in at least one test and one val set.")
    return folds





def print_splits(df, folds, n_folds):
 
    new_split_df = df.copy()    
    new_split_df = new_split_df.set_index('Ensembl_ID')
    new_split_df = new_split_df.drop(columns=new_split_df.columns)
    for fi in range(1, n_folds+1):
        train_ensid = folds[int(fi)-1]['train_idx'].tolist()
        valid_ensid = folds[int(fi)-1]['val_idx'].tolist()
        test_ensid = folds[int(fi)-1]['test_idx'].tolist()
        colname = 'fold'+str(fi)
        new_split_df[colname] = ''
        new_split_df.loc[train_ensid,colname] = 'train'
        new_split_df.loc[valid_ensid,colname] = 'valid'
        new_split_df.loc[test_ensid,colname] = 'test'
    print(new_split_df)
    new_split_df.to_csv("split.txt")



class Objective:
    def __init__(self, device, cell, expr_type, use_pretrained=False, n_enhancers=60, n_rnaFeat=9, 
                 n_extraFeat=2, batch_size=16, epochs = 10, saved_model_path="./trained_models/"):
        self.device = device
        self.cell = cell
        self.expr_type = expr_type
        self.use_pretrained = use_pretrained
        self.n_extraFeat = n_extraFeat
        self.batch_size = batch_size
        self.n_epoch = epochs
        self.saved_model_path = saved_model_path
        
        #today = datetime.now()   # Get date
        #datetime_str = today.strftime("%Y-%m-%d-%H")
        #split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
        #saved_model_path = './trained_models/{}/'.format(datetime_str)


    def get_fold_datasets(self, fi):
        split = self.splits[fi - 1]

        train_idx = self.get_valid_indices(split['train_idx'])
        val_idx = self.get_valid_indices(split['val_idx'])
        test_idx = self.get_valid_indices(split['test_idx'])

        train_ds = Subset(self.all_ds, train_idx)
        valid_ds = Subset(self.all_ds, val_idx)
        test_ds = Subset(self.all_ds, test_idx)
        return train_ds, valid_ds, test_ds

    def get_valid_indices(self, ens_ids):
        common = list(set(ens_ids).intersection(set(self.ensid_df.index), set(self.expr_df.index)))
        return self.ensid_df.loc[common]['idx']


    def train_one_fold(self, fold_i, trial_params):
        train_ds, valid_ds, _ = self.get_fold_datasets(fold_i)

        if self.use_pretrained:
            pretrained_convNet = enhancer_predictor_256bp()
            pt_model_name = '{}_seq2activityLog2_leaveChrOut_combinedRS_2bins_bs64_H3K27ac_adamW_erisxdl_r0'.format(cell)
            checkpoint = torch.load("./trained_models/pretrained_enhancer_encoder/{}_best_{}_checkpoint.pt".format('fold_' + str(fold_i), pt_model_name), map_location=torch.device(device))
            print('Loading pretrained model ...', pt_model_name)
            #model = EPInformer_v2(n_encoder=n_encoder, pre_trained_encoder=pretrained_convNet.encoder, n_enhancer=n_enhancers, out_dim=64, n_rnaFeat=n_rnaFeat, n_extraFeat=n_extraFeat, device=device).to(device)
            model = EPInformer_v2(
                n_encoder=trial_params['n_encoder'],
                pre_trained_encoder=pretrained_convNet.encoder,
                #out_dim=64,
                head=trial_params['head'],
                n_extraFeat=self.n_extraFeat,
                device=self.device,
                rna_method=trial_params['rna_method'],
                rna_transform=trial_params['rna_transform'],
                n_enhancer=60
            )
        else:
            model = EPInformer_v2(
                n_encoder=trial_params['n_encoder'],
                pre_trained_encoder=None,
                #out_dim=64,
                head=trial_params['head'],
                n_extraFeat=self.n_extraFeat,
                device=self.device,
                rna_method=trial_params['rna_method'],
                rna_transform=trial_params['rna_transform'],
                n_enhancer=60
            )

            #model = EPInformer_v2(n_encoder=n_encoder, pre_trained_encoder=None, n_enhancer=n_enhancers, out_dim=64, n_rnaFeat=n_rnaFeat, n_extraFeat=n_extraFeat, device=device).to(device)

        model = model.to(self.device)
        #model.name = model.name.replace('EPInformerV2', args.model_type) + '.' +  cell + '.' + expr_type
        #minus_val_r2 = utils.train(model, train_ds, valid_dataset=valid_ds, EPOCHS=n_epoch, model_name = model.name, fold_i=fi, batch_size=batch_size, device=device, saved_model_path=saved_model_path)
        val_r2 = utils.train_foropt(
            net=model,
            training_dataset=train_ds,
            valid_dataset=valid_ds,
            fold_i=fold_i,
            saved_model_path=self.saved_model_path,
            learning_rate=trial_params['learning_rate'],
            model_name=model.name,
            #batch_size=trial_params['batch_size'],
            batch_size=self.batch_size,
            device=self.device,
            EPOCHS=self.n_epoch,
            rna_method=trial_params['rna_method']
        )
        return val_r2

    def __call__(self, trial):
        trial_params = {
            #'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64]),
            #'useBN': trial.suggest_categorical('useBN', [True, False]), 
            #'useLN': trial.suggest_categorical('useLN', [True, False]),
            #'out_dim': trial.suggest_categorical("out_dim", [16, 32, 64]),
            'learning_rate': trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3]),
            'n_encoder': trial.suggest_int("n_encoder", 3, 4, 5),
            'head': trial.suggest_categorical("head", [4, 8, 16]),
            'rna_method': trial.suggest_categorical('rna_method', ['encoding', 'embedding', 'one-hot']),
            'rna_transform': trial.suggest_categorical('rna_transform', ['log10', 'sigmoid', 'tanh'])
            #'epochs': 10
        }
        
        rna_method = trial_params['rna_method']
        rna_transform = trial_params['rna_transform']
        
        #print(f"RNA Method: {rna_method}, Transform: {rna_transform}")

        EP_df = pd.read_csv(f'/home/witoslaw/data/diffTSS/{cell}_enhancer_gene_links_100kb.hg38.tsv', sep='\t')
        promoter_df = EP_df.groupby('TargetGeneEnsembl_ID', as_index = False)['chr'].first()
        promoter_df.rename(columns={'TargetGeneEnsembl_ID': 'Ensembl_ID'}, inplace=True)
        all_ds = utils.promoter_enhancer_dataset(data_folder= '/home/witoslaw/data/diffTSS/', expr_type=expr_type, cell_type=cell, n_extraFeat=n_extraFeat,
                                                 usePromoterSignal=True, n_enhancers=n_enhancers, hic_threshold=hic_threshold, distance_threshold=distance_threshold,
                                                 rna_method=rna_method, rna_transform=rna_transform)
        #print(f"all_ds[0]:\n{all_ds[0]}")
        
        ensid_list = [eid.decode() for eid in all_ds.data_h5['ensid'][:]]
        ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
        ensid_df['idx'] = np.arange(len(ensid_list))
        ensid_df = ensid_df.set_index('ensid')

        splits = generate_splits(promoter_df,'chr')
        #splits = generate_random_splits(promoter_df,'chr')
        #splits = load_pretrained_splits(promoter_df,'chr', './data/cvtable.txt')
        print_splits(promoter_df, splits, len(splits))
        
        self.all_ds = all_ds
        self.splits = splits
        self.ensid_df = ensid_df
        self.expr_df = self.all_ds.expr_df

        #fold_i = np.random.randint(1, len(self.splits) + 1)
        fold_i = 1
        val_r2 = self.train_one_fold(fold_i=fold_i, trial_params=trial_params)
        return val_r2






parser = argparse.ArgumentParser()
def list_of_strings(arg):
    return arg.split(',')
parser.add_argument('--cell', type=str, help='cell line (support K562 and GM12878)', choices=['K562', 'GM12878'])  
parser.add_argument("--fold", type=list_of_strings, help="test fold", default='all')
parser.add_argument("--model_type", type=str, help='EPInformer type', default='EPInformer-PE-Activity-HiC', choices=['EPInformer-PE', 'EPInformer-PE-Activity', 'EPInformer-PE-Activity-HiC'])  
parser.add_argument('--distance_threshold', type=int, help='max distance to TSS', default=100_000) 
parser.add_argument('--hic_threshold', type=int, help='hic loop thresold', default=-1) 
parser.add_argument('--expr_assay', type=str, help='expression_assay', choices=['CAGE', 'RNA'])
parser.add_argument('--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--n_interact_enc',type=int, help='layers of interaction encoder', default=3)
parser.add_argument('--epochs',type=int, help='training epochs', default=10)
parser.add_argument('--cuda', help='use cuda', action='store_true')
parser.add_argument('--use_pretrained_encoder', help='use pretrained sequence encoder', action='store_true')
parser.add_argument('--rna', help='option for rna encoding, embedding, or one-hot incorporation', choices=['encoding', 'embedding', 'one-hot', None], default=None)
parser.add_argument('--rna_transform', help='possible data transformations: log10, tanh, sigmoid', choices=['log10', 'tanh', 'sigmoid', None], default=None)
parser.add_argument('--host', help='optuna storage network host', default='localhost')
parser.add_argument('--port', help='optuna storage network port', default=0)

# example
# python train_EPInformer.py --cell K562  --model_type EPInformer-PE-Activity --expr_assay CAGE --use_pretrained_encoder --batch_size 16 --fold 1

##### parameter ######
args = parser.parse_args()

cell = args.cell

if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'
distance_threshold = args.distance_threshold
n_epoch = args.epochs
hic_threshold = args.hic_threshold
if hic_threshold == -1:
    hic_threshold = None

if args.model_type == 'EPInformer-PE': 
    n_extraFeat = 1
elif args.model_type == 'EPInformer-PE-Activity':
    n_extraFeat = 2
elif args.model_type == 'EPInformer-PE-Activity-HiC':
    n_extraFeat = 3

use_pretrained = args.use_pretrained_encoder

rna_method = args.rna
rna_transform = args.rna_transform

fold_list = args.fold 
n_encoder = args.n_interact_enc
batch_size = args.batch_size 
expr_type = args.expr_assay
n_enhancers = 60
postgres_host = args.host
postgres_port = args.port
pg_pass_file = '/home/'+os.environ["USER"]+'/postgres/config/postgres-password'

#################



objective = Objective(
    device=device,
    cell=cell,
    expr_type=expr_type,
    use_pretrained=use_pretrained,
    n_extraFeat=n_extraFeat,
    batch_size = batch_size,
    epochs = n_epoch,
    saved_model_path="./trained_models/optuna/"
)

# Create an Optuna study
study_name = "diffTSS_stranded_attnmask"
with open(pg_pass_file, 'r') as f:
    db_password = f.read().strip()
storage = optuna.storages.RDBStorage(url="postgresql://witoslaw:"+db_password+"@"+postgres_host+":"+str(postgres_port)+"/diffTSS")
study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage, load_if_exists=True)


# Start optimization
#study = optuna.create_study()
study.optimize(objective, n_trials=50)


# Get top N trials
#top_trials = study.best_trials[:5]
#for t in top_trials:
#    avg_r2 = 0
#    for fi in range(1, 6):
#        obj = Objective(..., fold_id=fi)
#        r2 = obj.train_one_fold(fi, t.params)
#        avg_r2 += r2
#    avg_r2 /= 5
#    print(f\"Trial {t.number}: Avg R2 over 5 folds = {avg_r2:.4f}\")

# Print best trial results
print("Best R²:", study.best_value)
print("Best hyperparameters:", study.best_params)

