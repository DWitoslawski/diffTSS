#!/usr/bin/env python
# coding: utf-8

# In[4]:


from EPInformer.models import EPInformer_v2, enhancer_predictor_256bp
from scripts.utils import prepare_input, prepare_hd5_input
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
      if pe_code_data.dtype != np.uint8:
          pe_code_data=pe_code_data.astype(np.uint8)
      h5_file.create_dataset('pe_code', data=pe_code_data, dtype="b1")

      # Create 'distance' dataset (example: 2D array with distances)
      #distance_data = np.random.rand(10, 5)  # Replace with real distances
      h5_file.create_dataset('distance', data=distance_data)

      # Create 'activity' dataset (example: 2D array for enhancer activity)
      #activity_data = np.random.rand(10, 5)  # Replace with real activity data
      h5_file.create_dataset('activity', data=activity_data)

      # Create 'hic' dataset (example: 2D array for Hi-C contact frequencies)
      #hic_data = np.random.rand(10, 5)  # Replace with real Hi-C data
      h5_file.create_dataset('hic', data=hic_data)
        
      h5_file.create_dataset('rna', data=rna_data)

  print(f"H5 file created at {file_path}")



enhancer_gene_k562_100kb = pd.read_csv('./data/K562_enhancer_gene_links_100kb.hg38.tsv', sep='\t')
gene_k562_tss = pd.read_csv('./data/ABC-multiTSS_nominated/K562/Neighborhoods/GeneList.txt', sep='\t')[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
gene_k562_tss['ENSID'] = gene_k562_tss['Ensembl_ID']


enhancer_gene_gm12878_100kb = pd.read_csv('./data/GM12878_enhancer_gene_links_100kb.hg38.tsv', sep='\t')
gene_gm12878_tss = pd.read_csv('./data/ABC-multiTSS_nominated/GM12878/Neighborhoods/GeneList.txt', sep='\t')[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
gene_gm12878_tss['ENSID'] = gene_gm12878_tss['Ensembl_ID']


enhancer_gene_k562_100kb_includeNoEnhancerGene = enhancer_gene_k562_100kb.merge(gene_k562_tss, left_on='TargetGeneEnsembl_ID', right_on='Ensembl_ID', how='right', suffixes=['', '_gene']).reset_index()
enhancer_gene_gm12878_100kb_includeNoEnhancerGene = enhancer_gene_gm12878_100kb.merge(gene_gm12878_tss, left_on='TargetGeneEnsembl_ID', right_on='Ensembl_ID', how='right', suffixes=['', '_gene']).reset_index()

gene_list = list(gene_k562_tss['ENSID'])


rna_df_K562 = pd.read_csv('./data/RNASeq_bw/K562.minus.ENCFF528VFJ.coverage.txt', header=None, sep='\t')
rna_df_GM12878 = pd.read_csv('./data/RNASeq_bw/GM12878.minus.ENCFF074SXQ.coverage.txt', header=None, sep='\t')


ensid_data, pe_code, distance_data, activity_data, hic_data, rna_data = prepare_hd5_input(enhancer_gene_k562_100kb_includeNoEnhancerGene, gene_k562_tss, gene_list, 'K562', num_features=3, rna_method='encoding', rna_df=rna_df_K562)

file_path = '/scratch/han_lab/dwito/EPInformer/K562_enhancer_promoter_encoding.rna_encoding.hg38.h5'
create_h5_data(file_path, ensid_data, pe_code, distance_data, activity_data, hic_data, rna_data)


#del ensid_data, pe_code, distance_data, activity_data, hic_data, rna_data


#ensid_data, pe_code, distance_data, activity_data, hic_data, rna_data = prepare_hd5_input(enhancer_gene_gm12878_100kb_includeNoEnhancerGene, gene_gm12878_tss, gene_list, 'GM12878', num_features=3, rna_method='encoding', rna_df=rna_df_GM12878)

#file_path = '/scratch/han_lab/dwito/EPInformer/GM12878_enhancer_promoter_encoding.rna_encoding.hg38.h5'
#create_h5_data(file_path, ensid_data, pe_code, distance_data, activity_data, hic_data, rna_data)
