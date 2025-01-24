#!/usr/bin/env python
# coding: utf-8

# In[4]:


from EPInformer.models import EPInformer_v2, enhancer_predictor_256bp
from scripts.utils import prepare_input
import scripts.utils_forTraining as train
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import torch
from torch.utils.data import Subset, Dataset


# In[ ]:


# Download K562 training data from Zenodo
get_ipython().system('wget https://zenodo.org/records/12738705/files/K562_DNase_ENCFF257HEE_2kb_4DNFITUOMFUQ_enhancer_promoter_encoding.h5.zip -P ./data/')
get_ipython().system('wget https://zenodo.org/records/12738705/files/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated.zip -P ./data/')
get_ipython().system('unzip -o -qq ./data/K562_DNase_ENCFF257HEE_2kb_4DNFITUOMFUQ_enhancer_promoter_encoding.h5.zip -d ./data/')
get_ipython().system('unzip -o -qq ./data/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated.zip -d ./data/')


# In[ ]:


# Download GM2878 training data from Zenodo
get_ipython().system('wget https://zenodo.org/records/12738705/files/GM12878_DNase_ENCFF020WZB_2kb_4DNFI1UEG1HD_promoter_enhancer_encoding.h5.zip -P ./data/')
get_ipython().system('wget https://zenodo.org/records/12738705/files/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated.zip -P ./data/')
get_ipython().system('unzip -o -qq ./data/GM12878_DNase_ENCFF020WZB_2kb_4DNFI1UEG1HD_promoter_enhancer_encoding.h5.zip -d ./data/')
get_ipython().system('unzip -o -qq ./data/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated.zip -d ./data/')


# ## Reproduce prediction results

# In[3]:


# 12-fold cross-validation split
split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)


# In[5]:


cell = 'K562'
distance_threshold = 100_000
n_enhancers = 60
device = 'cuda'
# num_feature == 1: distance; num_feature == 2: distance + enhancer activity; num_feature == 3: distance + enhancer activity + hic contacts
n_extraFeat = 3
batch_size = 16
expr_type = 'RNA'
prediction_res = []
for fi in range(1, 13):
    print("-"*10, 'fold', fi, '-'*10)
    fold_i = 'fold_' + str(fi)

    train_ensid = split_df[split_df[fold_i] == 'train'].index
    valid_ensid = split_df[split_df[fold_i] == 'valid'].index
    test_ensid = split_df[split_df[fold_i] == 'test'].index

    all_ds = train.promoter_enhancer_dataset(expr_type=expr_type, cell_type=cell, n_extraFeat=n_extraFeat, usePromoterSignal=True, n_enhancers=n_enhancers, distance_threshold=distance_threshold, data_folder = './data/')
    ensid_list = [eid.decode() for eid in all_ds.data_h5['ensid'][:]]
    ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
    ensid_df['idx'] = np.arange(len(ensid_list))
    ensid_df = ensid_df.set_index('ensid')
    train_idx = ensid_df.loc[train_ensid]['idx']
    valid_idx = ensid_df.loc[valid_ensid]['idx']

    test_idx = ensid_df.loc[test_ensid]['idx']

    train_ds = Subset(all_ds, train_idx)
    valid_ds = Subset(all_ds, valid_idx)
    test_ds = Subset(all_ds, test_idx)

    model = EPInformer_v2(n_encoder=3, n_enhancer=n_enhancers, out_dim=64, n_extraFeat=3, device=device)
    model = model.to(device)
    # model_name= 'tunedEPInformerV2.preTrainedConv.4base.64dim.3Trans.4head.TrueBN.TrueLN.TrueFeat.3extraFeat.60enh.K562.rmEnhNone.bs16.seq_feat_dist.DNaseH.distanceDist100k.hic0.len2k.distance.{}'.format(expr_type)
    checkpoint = torch.load("./trained_models/EPInformer_PE_Activity_HiC/K562/fold_{}_EPInformer_PE_Activity_HiC_{}_K562_checkpoint.pt".format(fi, expr_type))
    # model_path = "/content/drive/MyDrive/EPInformer/EPInformer_activity/models_allInOne/fold_{}_best_{}_checkpoint.pt".format(fi, model_name)
    # checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    test_df = train.test(model, test_ds, fold_i=fi, batch_size=batch_size, device=device)
    prediction_res.append(test_df)
prediction_res = pd.concat(prediction_res)


# In[6]:


pearsonR, pv = stats.pearsonr(prediction_res['Pred'], prediction_res['actual'])
print('PearsonR of 12-fold cross-validation predicitons', pearsonR, 'Total genes:', len(prediction_res))


# In[7]:


plt.figure(figsize=(6,6))
ax = sns.jointplot(
    data=prediction_res,
    x="Pred",
    y="actual",
    kind = 'scatter',
    joint_kws={'marker':'o', 's':10, 'alpha':0.1, 'linewidth':0},
    marginal_kws={'bins':20, 'element':'step', 'kde':True, 'linewidth':0},
)
ax.plot_joint(sns.regplot, color="r", scatter=False, line_kws={"color": "orange", 'linestyle':'dashed'})
plt.title('pearsonR = {:.3f}'.format(pearsonR))
plt.ylabel('K562 RNA-seq expression')
plt.xlabel('12-fold cross-chromosome predictions')
plt.tight_layout()


# In[10]:


cell = 'GM12878'
distance_threshold = 100_000
n_enhancers = 60
device = 'cuda'
# num_feature == 1: distance; num_feature == 2: distance + enhancer activity; num_feature == 3: distance + enhancer activity + hic contacts
n_extraFeat = 3
batch_size = 16
expr_type = 'RNA'
prediction_res = []
for fi in range(1, 13):
    print("-"*10, 'fold', fi, '-'*10)
    fold_i = 'fold_' + str(fi)

    train_ensid = split_df[split_df[fold_i] == 'train'].index
    valid_ensid = split_df[split_df[fold_i] == 'valid'].index
    test_ensid = split_df[split_df[fold_i] == 'test'].index

    all_ds = train.promoter_enhancer_dataset(expr_type=expr_type, cell_type=cell, n_extraFeat=n_extraFeat, n_enhancers=n_enhancers, distance_threshold=distance_threshold)
    ensid_list = [eid.decode() for eid in all_ds.data_h5['ensid'][:]]
    ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
    ensid_df['idx'] = np.arange(len(ensid_list))
    ensid_df = ensid_df.set_index('ensid')
    train_idx = ensid_df.loc[train_ensid]['idx']
    valid_idx = ensid_df.loc[valid_ensid]['idx']
    test_idx = ensid_df.loc[test_ensid]['idx']

    train_ds = Subset(all_ds, train_idx)
    valid_ds = Subset(all_ds, valid_idx)
    test_ds = Subset(all_ds, test_idx)

    model = EPInformer_v2(n_encoder=3, n_enhancer=n_enhancers, out_dim=64, n_extraFeat=n_extraFeat, device=device)
    checkpoint = torch.load("./trained_models/EPInformer_PE_Activity_HiC/GM12878/fold_{}_EPInformer_PE_Activity_HiC_{}_GM12878_checkpoint.pt".format(fi, expr_type))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    test_df = train.test(model, test_ds, fold_i=fi, batch_size=batch_size, device=device)
    prediction_res.append(test_df)
prediction_res = pd.concat(prediction_res)


# In[11]:


pearsonR, pv = stats.pearsonr(prediction_res['Pred'], prediction_res['actual'])
print('PearsonR of 12-fold cross-validation predicitons', pearsonR, 'Total genes:', len(prediction_res))


# In[12]:


plt.figure(figsize=(6,6))
ax = sns.jointplot(
    data=prediction_res,
    x="Pred",
    y="actual",
    kind = 'scatter',
    joint_kws={'marker':'o', 's':10, 'alpha':0.1, 'linewidth':0},
    marginal_kws={'bins':20, 'element':'step', 'kde':True, 'linewidth':0},
)
ax.plot_joint(sns.regplot, color="r", scatter=False, line_kws={"color": "orange", 'linestyle':'dashed'})
plt.title('pearsonR = {:.3f}'.format(pearsonR))
plt.ylabel('GM12878 RNA-seq expression')
plt.xlabel('12-fold cross-chromosome predictions')
plt.tight_layout()


# ## Predict gene expression given promoter-enhancer sequence

# In[7]:


# Donwload reference genome
get_ipython().system('wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz  -P ./data/')
get_ipython().system('gunzip ./data/hg38.fa.gz')


# In[5]:


# Load ABC enhancer-gene data
enhancer_gene_k562 = pd.read_csv('./data/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated/Gene-enhancer links/EnhancerPredictionsAllPutative.txt.gz', sep='\t')
# Select the gene-enhancer links within 100kb to the TSS of target gene and remove the promoter element
enhancer_gene_k562_100kb = enhancer_gene_k562[(enhancer_gene_k562['distance']<=100_000)&(enhancer_gene_k562['distance']>1000)].reset_index()
enhancer_gene_k562_100kb.to_csv('./data/K562_enhancer_gene_links_100kb.tsv', index=False, sep='\t')


# In[6]:


enhancer_gene_k562_100kb = pd.read_csv('./data/K562_enhancer_gene_links_100kb.tsv', sep='\t')
gene_tss = pd.read_csv('./data/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt', sep='\t')[['name', 'chr', 'tss', 'strand']]
data_split = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv')
gene_tss = gene_tss.merge(data_split[['ENSID', 'Gene name']], left_on='name', right_on='ENSID').drop(columns='name')
enhancer_gene_k562_100kb_includeNoEnhancerGene = enhancer_gene_k562_100kb.merge(gene_tss, left_on='TargetGene', right_on='ENSID', how='right', suffixes=['', '_gene']).reset_index()


# In[24]:


enhancer_gene_k562_100kb_includeNoEnhancerGene.head(3)


# In[18]:


gene_info = data_split[data_split['fold_1'] == 'test'][['ENSID', 'Gene name']].head(16).reset_index(drop=True)
gene_list = list(gene_info['ENSID'])
# encode gene-enhancer links for EPInformer
# num_feature == 1: distance; num_feature == 2: distance + enhancer activity; num_feature == 3: distance + enhancer activity + hic contacts
device = 'cpu'
PE_codes, PE_feats, mRNA_feats, PE_pairs = prepare_input(enhancer_gene_k562_100kb_includeNoEnhancerGene, gene_list, 'K562', num_features=2)
PE_codes = torch.from_numpy(PE_codes).float().to(device)
PE_feats = torch.from_numpy(PE_feats).float().to(device)
mRNA_feats = torch.from_numpy(mRNA_feats).float().to(device)
print(PE_codes.shape, PE_feats.shape, mRNA_feats.shape)


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


gene_info


# In[ ]:




