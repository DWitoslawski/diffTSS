#!/usr/bin/env python
# coding: utf-8

# In[4]:


from EPInformer.models import EPInformer_v2, enhancer_predictor_256bp
from scripts.utils import prepare_input, prepare_input_diff, prepare_hd5_input_diff
import scripts.utils_forTraining as train
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import torch
from torch.utils.data import Subset, Dataset
import h5py



def create_h5_data(file_path, ensid_data, pe_code_data, distance_data, activity_data, hic_data):

  # Create a new H5 file
  with h5py.File(file_path, 'w') as h5_file:
      # Create 'ensid' dataset (an array of encoded strings)
      #ensid_data = [f"ENSG{100000 + i}" for i in range(10)]  # Example ENSIDs
      encoded_ensid_data = [s.encode() for s in ensid_data]  # Encode strings as bytes
      h5_file.create_dataset('ensid', data=np.array(encoded_ensid_data, dtype='S'))

      # Create 'pe_code' dataset (example: 2D array of integers)
      #pe_code_data = np.random.randint(0, 100, size=(10, 20))  # Replace with real data
      h5_file.create_dataset('pe_code', data=pe_code_data)

      # Create 'distance' dataset (example: 2D array with distances)
      #distance_data = np.random.rand(10, 5)  # Replace with real distances
      h5_file.create_dataset('distance', data=distance_data)

      # Create 'activity' dataset (example: 2D array for enhancer activity)
      #activity_data = np.random.rand(10, 5)  # Replace with real activity data
      h5_file.create_dataset('activity', data=activity_data)

      # Create 'hic' dataset (example: 2D array for Hi-C contact frequencies)
      #hic_data = np.random.rand(10, 5)  # Replace with real Hi-C data
      h5_file.create_dataset('hic', data=hic_data)

  print(f"H5 file created at {file_path}")



# ## Predict gene expression given promoter-enhancer sequence

# In[7]:


# Donwload reference genome
#get_ipython().system('wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz  -P ./data/')
#get_ipython().system('gunzip ./data/hg38.fa.gz')


# In[5]:


# Load ABC enhancer-gene data
#enhancer_gene_k562 = pd.read_csv('./data/K562_GM12878_ABC_nominated/Gene-enhancer links/K562.AllPutative.txt', sep='\t')
# Select the gene-enhancer links within 100kb to the TSS of target gene and remove the promoter element
#enhancer_gene_k562_100kb = enhancer_gene_k562[(enhancer_gene_k562['distance']<=100_000)&(enhancer_gene_k562['distance']>1000)].reset_index()
#enhancer_gene_k562_100kb.to_csv('./data/K562_enhancer_gene_links_100kb.tsv', index=False, sep='\t')

#enhancer_gene_gm12878 = pd.read_csv('./data/K562_GM12878_ABC_nominated/Gene-enhancer links/GM12878.AllPutative.txt', sep='\t')
# Select the gene-enhancer links within 100kb to the TSS of target gene and remove the promoter element
#enhancer_gene_gm12878_100kb = enhancer_gene_gm12878[(enhancer_gene_gm12878['distance']<=100_000)&(enhancer_gene_gm12878['distance']>1000)].reset_index()
#enhancer_gene_gm12878_100kb.to_csv('./data/GM12878_enhancer_gene_links_100kb.tsv', index=False, sep='\t')


# In[6]:


enhancer_gene_k562_100kb = pd.read_csv('./data/K562_enhancer_gene_links_100kb.tsv', sep='\t')
gene_k562_tss = pd.read_csv('./data/K562_GM12878_ABC_nominated/Neighborhoods/K562.GeneList.txt', sep='\t')[['name', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
# todo: data_split file need to be updated to common GeneList.txt 
data_split = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv')
gene_k562_tss = gene_k562_tss.merge(data_split[['ENSID', 'Gene name']], left_on='name', right_on='Gene name').drop(columns='name')

enhancer_gene_gm12878_100kb = pd.read_csv('./data/GM12878_enhancer_gene_links_100kb.tsv', sep='\t')
gene_gm12878_tss = pd.read_csv('./data/K562_GM12878_ABC_nominated/Neighborhoods/GM12878.GeneList.txt', sep='\t')[['name', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
# todo: data_split file need to be updated to common GeneList.txt 
data_split = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv')
gene_gm12878_tss = gene_gm12878_tss.merge(data_split[['ENSID', 'Gene name']], left_on='name', right_on='Gene name').drop(columns='name')


#enhancer_gene_k562_100kb_includeNoEnhancerGene = enhancer_gene_k562_100kb.merge(gene_k562_tss, left_on='TargetGene', right_on='Gene name', how='right', suffixes=['', '_gene']).reset_index()
#enhancer_gene_gm12878_100kb_includeNoEnhancerGene = enhancer_gene_gm12878_100kb.merge(gene_gm12878_tss, left_on='TargetGene', right_on='Gene name', how='right', suffixes=['', '_gene']).reset_index()
enhancer_gene_k562_100kb_inner = enhancer_gene_k562_100kb.merge(gene_k562_tss, left_on='TargetGene', right_on='Gene name', how='inner', suffixes=['', '_gene']).reset_index()
enhancer_gene_gm12878_100kb_inner = enhancer_gene_gm12878_100kb.merge(gene_gm12878_tss, left_on='TargetGene', right_on='Gene name', how='inner', suffixes=['', '_gene']).reset_index()



# In[24]:


enhancer_gene_k562_100kb_inner.head(3)
enhancer_gene_gm12878_100kb_inner.head(3)


# In[18]:


gene_info = data_split[data_split['fold_1'] == 'test'][['ENSID', 'Gene name']].head(16).reset_index(drop=True)
gene_list = list(set(gene_info['ENSID']).intersection(set(enhancer_gene_gm12878_100kb_inner['ENSID'])))
gene_info.set_index('ENSID',inplace=True)
gene_info = gene_info.loc[gene_list]

gene_list = list(set(enhancer_gene_k562_100kb_inner['ENSID']).intersection(set(enhancer_gene_gm12878_100kb_inner['ENSID'])))
# encode gene-enhancer links for EPInformer
# num_feature == 1: distance; num_feature == 2: distance + enhancer activity; num_feature == 3: distance + enhancer activity + hic contacts
device = 'cpu'
#PE_codes, PE_feats, mRNA_feats, PE_pairs = prepare_input_diff(enhancer_gene_k562_100kb_inner, enhancer_gene_gm12878_100kb_inner, gene_k562_tss, gene_gm12878_tss, gene_list, 'K562', num_features=2)
#PE_codes = torch.from_numpy(PE_codes).float().to(device)
#PE_feats = torch.from_numpy(PE_feats).float().to(device)
#mRNA_feats = torch.from_numpy(mRNA_feats).float().to(device)
#print(PE_codes.shape, PE_feats.shape, mRNA_feats.shape)


# File path to save the H5 file
# PE_code_list, PE_distance_list, PE_activity_list, PE_contact_list
ensid_data, pe_code, distance_data, activity_data, hic_data = prepare_hd5_input_diff(enhancer_gene_k562_100kb_inner, enhancer_gene_gm12878_100kb_inner, gene_k562_tss, gene_gm12878_tss, gene_list, 'K562', num_features=2)
breakpoint()
file_path = './data/K562_GM12878_enhancer_promoter_encoding.h5'
create_h5_data(file_path, ensid_data, pe_code, distance_data, activity_data, hic_data)

# In[21]:


# Load pre-trained EPInformer-PE-Activity (CAGE-seq)
expr_type = 'CAGE'
model = EPInformer_v2(n_encoder=3, n_enhancer=60, out_dim=64, n_extraFeat=2, device=device)
model_path = './trained_models/EPInformer_PE_Activity/K562/fold_1_EPInformer_PE_Activity_{}_K562_checkpoint.pt'.format(expr_type)
checkpoint = torch.load(model_path, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)


# In[22]:


model.eval()
with torch.no_grad():
    pred_expr, _ = model(PE_codes, mRNA_feats, PE_feats)
    pred_expr = pred_expr.numpy().squeeze()
gene_info['predicted_expr'] = pred_expr
gene_info['expr_type'] = expr_type


# In[23]:


print(gene_info)


# In[ ]:




