import sys
import shutil
import argparse

from datetime import datetime
import os
import json
import scripts.utils_forTraining as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from EPInformer.models import EPInformer_v2, enhancer_predictor_256bp
from scipy import stats
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, EarlyStoppingCallback


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


def compute_metrics(eval_pred):
    """Computes regression metrics for evaluation."""
    predictions, labels = eval_pred
    # The model outputs a single value, so we squeeze it
    predictions = predictions.squeeze()
    
    predictions_pt = torch.from_numpy(predictions)
    labels_pt = torch.from_numpy(labels)
    
    loss_fn = nn.SmoothL1Loss()
    
    # Calculate metrics
    pearson_corr, _ = stats.pearsonr(predictions, labels)
    spearman_corr, _ = stats.spearmanr(predictions, labels)
    mse = mean_squared_error(predictions, labels)
    smooth_l1_loss = loss_fn(predictions_pt, labels_pt).item()
    
    print(f"SmoothL1Loss: {smooth_l1_loss}")
    
    return {
        "pearsonr": pearson_corr,
        "spearmanr": spearman_corr,
        "mse": mse,
        "smooth_l1_loss": smooth_l1_loss
    }    


def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")
    

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument('--cell', type=str, help='cell line (support K562 and GM12878)', choices=['K562', 'GM12878'])  
    parser.add_argument("--fold", type=list_of_strings, help="test fold", default='all')
    parser.add_argument('--expr_assay', type=str, help='expression_assay', choices=['CAGE', 'RNA'])
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--epochs',type=int, help='training epochs', default=3)
    parser.add_argument('--model_size',type=str, help='NTv2 model: 50m or 500m', choices=['50m', '500m'], default='50m')

    # example
    # python train_EPInformer.py --cell K562 --expr_assay CAGE --batch_size 16 

    ##### parameter ######
    args = parser.parse_args()
    
    cell = args.cell
    
    n_epoch = args.epochs

    fold_list = args.fold 
    batch_size = args.batch_size 
    expr_type = args.expr_assay
    model_size = args.model_size
    
    model_path = f"/home/witoslaw/dna_language_models/nucleotide-transformer-v2-{model_size}-multi-species/"

    #################

    today = datetime.now()   # Get date

    datetime_str = today.strftime("%Y-%m-%d-%H")
    #split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
    saved_model_path = './trained_models/NT-v2/{}/'.format(datetime_str)

    EP_df = pd.read_csv(f'/home/witoslaw/data/diffTSS/data/{cell}_enhancer_gene_links_100kb.hg38.tsv', sep='\t')
    promoter_df = EP_df.groupby('TargetGeneEnsembl_ID', as_index = False)['chr'].first()
    promoter_df.rename(columns={'TargetGeneEnsembl_ID': 'Ensembl_ID'}, inplace=True)
    
    data_folder = '/home/witoslaw/data/diffTSS/data/'
    
    dataset = load_dataset("json", data_files=f"{data_folder}{cell}_promoter_seq_CAGE.json", field='data', split='train')    
    dataset = dataset.rename_column('label', 'labels')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset.set_format("torch")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_labels = 1
    config.problem_type = "regression"
    
    with open('/home/witoslaw/data/diffTSS/data/K562_promoter_seq_CAGE.json') as f:
        ensid_list = json.load(f)
    
    ensid_list = ensid_list['ensid']
    ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
    ensid_df['idx'] = np.arange(len(ensid_list))
    ensid_df = ensid_df.set_index('ensid')

    splits = generate_splits(promoter_df,'chr')
    print_splits(promoter_df, splits)


    if 'all' in fold_list:
        fold_list = list(range(1, 13))
    else:
        fold_list = fold_list
    for fi in fold_list:
        print("-"*10, 'fold', fi, '-'*10)
        fold_i = 'fold_' + str(fi)

        train_ensid = splits[int(fi)-1]['train_idx'].tolist()
        valid_ensid = splits[int(fi)-1]['val_idx'].tolist()
        test_ensid = splits[int(fi)-1]['test_idx'].tolist()

        train_common_ensid = list(set(train_ensid).intersection(set(ensid_df.index)))
        train_idx = ensid_df.loc[train_common_ensid]['idx']
        valid_common_ensid = list(set(valid_ensid).intersection(set(ensid_df.index)))
        valid_idx = ensid_df.loc[valid_common_ensid]['idx']

        test_common_ensid = list(set(test_ensid).intersection(set(ensid_df.index)))
        test_idx = ensid_df.loc[test_common_ensid]['idx']

        train_test_valid_dataset = DatasetDict({
        'train': tokenized_dataset.select(train_idx),
        'test':  tokenized_dataset.select(test_idx),
        'valid': tokenized_dataset.select(valid_idx)})
        
        train_test_valid_dataset.set_format("torch")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            config=config,
            trust_remote_code=True
        )
        
        # Freeze the pretrained body
        # The classifier head parameters are trainable by default
        for param in model.base_model.parameters():
            param.requires_grad = False

            
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=6,  # Stop if no improvement for 6 evaluations
            #early_stopping_threshold=0.01 # Require at least 0.01 improvement
        )
        
        training_args = TrainingArguments(
            output_dir=os.path.join(saved_model_path, f'{fold_i}_results'),
            learning_rate=1e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=n_epoch,
            weight_decay=0.01,
            metric_for_best_model="eval_loss",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_test_valid_dataset["train"],
            eval_dataset=train_test_valid_dataset["valid"],
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback],
        )
        
        trainer.train()
        
        log_path = os.path.join(saved_model_path, f"{fold_i}_results/log.csv")
        log_history = pd.DataFrame(trainer.state.log_history)
        log_history.to_csv(log_path)
        
        results = trainer.evaluate(train_test_valid_dataset["test"])
        print(results)
        
        
        final_save_path = os.path.join(saved_model_path, f'/{cell}_{fold_i}_fine_tuned_model')
        model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        required_file = os.path.join(model_path, "/modeling_esm.py")

        source_file = os.path.join(model_path, required_file)
        destination_file = os.path.join(final_save_path, required_file)

        if os.path.exists(source_file):
            print(f"Copying {required_file} to {final_save_path}...")
            shutil.copyfile(required_file, destination_file)
        else:
            print(f"Warning: Could not find required file {required_file} to copy.")

        print(f"Model for fold {fi} saved correctly with custom code at {final_save_path}")