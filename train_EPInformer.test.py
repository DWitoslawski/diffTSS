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
from sklearn.model_selection import GroupKFold


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


def print_splits(df, folds):
 
    new_split_df = df.copy()    
    new_split_df = new_split_df.set_index('Ensembl_ID')
    new_split_df = new_split_df.drop(columns=new_split_df.columns)
    for fi in range(1, 13):
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



parser = argparse.ArgumentParser()
def list_of_strings(arg):
    return arg.split(',')
parser.add_argument('--cell', type=str, help='cell line (support K562 and GM12878)', choices=['K562', 'GM12878'])  
parser.add_argument("--fold", type=list_of_strings, help="test fold", default='all')
parser.add_argument("--model_type", type=str, help='EPInformer type', default='EPInformer-PE-Activity', choices=['EPInformer-PE', 'EPInformer-PE-Activity', 'EPInformer-PE-Activity-HiC'])  
parser.add_argument('--distance_threshold', type=int, help='max distance to TSS', default=100_000) 
parser.add_argument('--hic_threshold', type=int, help='hic loop thresold', default=-1) 
parser.add_argument('--expr_assay', type=str, help='expression_assay', choices=['CAGE', 'RNA'])
parser.add_argument('--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--n_interact_enc',type=int, help='layers of interaction encoder', default=3)
parser.add_argument('--epochs',type=int, help='training epochs', default=100)
parser.add_argument('--cuda', help='use cuda', action='store_true')
parser.add_argument('--use_pretrained_encoder', help='use pretrained sequence encoder', action='store_true')
parser.add_argument('--rna_encoding', help='input rna into sequence encoder', action='store_true')

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
rna_encoding = args.rna_encoding
fold_list = args.fold 
n_encoder = args.n_interact_enc
batch_size = args.batch_size 
expr_type = args.expr_assay
n_enhancers = 60

#################

today = datetime.now()   # Get date

datetime_str = today.strftime("%Y-%m-%d-%H")
#split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
saved_model_path = './trained_models/{}/'.format(datetime_str)

EP_df = pd.read_csv('./data/' + 'K562_enhancer_gene_links_100kb.hg38.tsv', sep='\t')
promoter_df = EP_df.groupby('TargetGeneEnsembl_ID', as_index = False)['chr'].first()
promoter_df.rename(columns={'TargetGeneEnsembl_ID': 'Ensembl_ID'}, inplace=True)
all_ds = utils.promoter_enhancer_dataset(data_folder= './data/', expr_type=expr_type, cell_type=cell, n_extraFeat=n_extraFeat, usePromoterSignal=True, n_enhancers=n_enhancers, hic_threshold=hic_threshold, distance_threshold=distance_threshold, rna_encoding=rna_encoding)
ensid_list = [eid.decode() for eid in all_ds.data_h5['ensid'][:]]
ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
ensid_df['idx'] = np.arange(len(ensid_list))
ensid_df = ensid_df.set_index('ensid')

if rna_encoding:
    rna_enc = 'rna_encoding'
else:
    rna_enc = ''

EP_df.to_csv(f'./testing/EP_{rna_enc}_df.tsv', sep='\t', index=False)
promoter_df.to_csv(f'./testing/promoter_{rna_enc}_df.tsv', sep='\t', index=False)
ensid_df.to_csv(f'./testing/ensid_{rna_enc}_df.tsv', sep='\t', index=False)
all_ds.promoter_df.to_csv(f'./testing/all_ds_promoter_{rna_enc}_df.tsv', sep='\t', index=False)
print(f'{all_ds.data_h5["ensid"][0]} with promoter code (length {len(all_ds.data_h5["pe_code"][0][:1].squeeze())}: \n{all_ds.data_h5["pe_code"][0][:1].squeeze()}')
file = open(f'./testing/all_ds_{rna_enc}_.txt', 'w')
print(all_ds[0], file=file)
file.close()
