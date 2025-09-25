from kipoiseq import Interval
import pyfaidx
import kipoiseq
import argparse
import numpy as np
import pandas as pd
import pyranges as pr
# from Bio.Seq import Seq
from tqdm import tqdm
import os
import h5py
import glob
from multiprocessing import Pool
from scripts.utils import FastaStringExtractor, one_hot_encode


def rc_dna(seq):
    """
    Reverse complement the DNA sequence
    >>> assert rc_seq("TATCG") == "CGATA"
    >>> assert rc_seq("tatcg") == "cgata"
    """
    rc_hash = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N",
        "a": "t",
        "t": "a",
        "c": "g",
        "g": "c",
        "n": "n",
    }
    return "".join([rc_hash[s] for s in reversed(seq)])


def create_h5_data(file_path, ensid_data, pe_code_data, distance_data, activity_data, hic_data, rna_data):

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




def process_gene(gene):
    #print(globals())
    gene_df = gene_enhancer_table[gene_enhancer_table['ENSID'] == gene]
    gene_rna_df_list = []
    if rna_method is not None:
        for rna_df in rna_df_list:
            if rna_method == 'encoding' or rna_method == 'one-hot':
                gene_rna_df_list.append(rna_df[rna_df[3] == gene])
            if rna_method == 'embedding':
                gene_rna_df = np.array(rna_df.loc[gene])
            
            '''
			pe_code, enhancer_activity, enhancer_distance, enhancer_contact, gene_name, gene_element_pair, rna_df
            PE_code, activity_list, distance_list, contact_list, gene_name, PE_links, gene_rna_df = encode_promoter_enhancer_links(gene_df, max_seq_len=2000, max_n_enhancer=60, max_distanceToTSS=100_000, add_flanking=False, rna_method=rna_method, rna_df=gene_rna_df)

        else:
            PE_code, activity_list, distance_list, contact_list, gene_name, PE_links = encode_promoter_enhancer_links(gene_df, max_seq_len=2000, max_n_enhancer=60, max_distanceToTSS=100_000, add_flanking=False, rna_method=rna_method, rna_df=gene_rna_df)
            '''
    
    fasta_path = './data/hg38.fa'
    fasta_extractor = FastaStringExtractor(fasta_path)
    gene_pe = gene_df.sort_values(by='distance')
    row_0 = gene_pe.iloc[0]
    gene_ensid = row_0['TargetGeneEnsembl_ID']
    gene_name = row_0['TargetGene']
    gene_tss = row_0['TargetGeneTSS']
    gene_strand = row_0['strand']
    chrom = row_0['chr']
    if row_0['TargetGeneTSS'] != row_0['TargetGeneTSS']:
        gene_tss = row_0['tss']
        gene_name = row_0['name_gene']
        chrom = row_0['chr_gene']
    target_interval = kipoiseq.Interval(chrom, int(gene_tss-max_seq_len/2), int(gene_tss+max_seq_len/2))
    promoter_seq = fasta_extractor.extract(target_interval)
    if gene_strand == '-':
        promoter_seq = rc_dna(promoter_seq)
    promoter_code = one_hot_encode(promoter_seq)
    if rna_method == 'encoding' or rna_method == 'one-hot':
        rna_signal_list = []
        for gene_rna_df in gene_rna_df_list:
            gene_rna_df = gene_rna_df[(gene_rna_df[7] >= target_interval.start) & (gene_rna_df[8] <= target_interval.end)]
            rna_signal = gene_rna_df[[9]]
            new_index = gene_rna_df[7].values - target_interval.start
            rna_signal = rna_signal.set_index(new_index).reindex(list(range(0,max_seq_len)), fill_value=0)
            if gene_strand == '-':
                rna_signal = rna_signal[::-1]
            rna_signal = np.array(rna_signal).flatten()
            rna_signal_list.append(rna_signal)
        rna_signal_list = np.array(rna_signal_list)
        #rna_signal = rna_signal.apply(lambda x: np.log10(x + 1))
        #promoter_code = np.concatenate((promoter_code, rna_signal), axis=1)
    if rna_method == 'embedding':
        gene_rna_df = np.concatenate([gene_rna_df.reshape(1, 125), np.zeros([60, 125])])
    enhancers_code = np.zeros((max_n_enhancer, max_seq_len, 4))
    enhancer_activity = np.zeros(max_n_enhancer)
    enhancer_distance = np.zeros(max_n_enhancer)
    enhancer_contact = np.zeros(max_n_enhancer)
    # set distance threshold
    gene_pe = gene_pe[(gene_pe['distance'] > max_seq_len/2)&(gene_pe['distance'] <= max_distanceToTSS)]
    e_i = 0
    gene_element_pair = []
    for idx, row in gene_pe.iterrows():
        if row['TargetGene'] != row['TargetGene']:
            break
        if pd.isna(row['start']):
            continue
        if e_i >= max_n_enhancer:
            break
        enhancer_start = int(row['start'])
        enhancer_end = int(row['end'])
        enhancer_center = int((row['start'] + row['end'])/2)
        enhancer_len = enhancer_end - enhancer_start
        # put sequence at the center
        if add_flanking:
            enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_center-int(max_seq_len/2), enhancer_center+int(max_seq_len/2))
            enhancers_code[e_i][:] = one_hot_encode(fasta_extractor.extract(enhancer_target_interval))
        else:
            # enhancers_signal = np.zeros((max_n_enhancer, max_seq_len))
            if enhancer_len > max_seq_len:
                enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_center-int(max_seq_len/2), enhancer_center+int(max_seq_len/2))
                enhancers_code[e_i][:] = one_hot_encode(fasta_extractor.extract(enhancer_target_interval))
            else:
                code_start = int(max_seq_len/2)-int(enhancer_len/2)
                enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_start, enhancer_end)
                enhancers_code[e_i][code_start:code_start+enhancer_len] = one_hot_encode(fasta_extractor.extract(enhancer_target_interval))
        # put sequence from the start
        enhancer_activity[e_i] = row['activity_base']
        enhancer_distance[e_i] = row['distance']
        enhancer_contact[e_i] = row['hic_contact']
        gene_element_pair.append([gene_name, row['name']])
        e_i += 1
    # print(promoter_signals.shape, enhancers_signal.shape)
    '''if rna_method == 'encoding':
        enhancers_code = np.concatenate((enhancers_code, np.zeros((max_n_enhancer, max_seq_len, 1))), axis=2)
        pe_code = np.concatenate([promoter_code[np.newaxis,:], enhancers_code], axis=0, dtype=np.float32)'''
    pe_code = np.concatenate([promoter_code[np.newaxis,:], enhancers_code], axis=0)
    gene_element_pair = pd.DataFrame(gene_element_pair, columns=['gene', 'element'])
    #if rna_method is not None:
    #    return pe_code, enhancer_activity, enhancer_distance, enhancer_contact, gene_name, gene_element_pair, rna_df
    #return pe_code, enhancer_activity, enhancer_distance, enhancer_contact, gene_name, gene_element_pair
            
        
    enhancer_contact = np.concatenate([[0], enhancer_contact])
    enhancer_distance = np.concatenate([[0], enhancer_distance/1000])
    enhancer_activity = np.concatenate([[0], enhancer_activity])
    # activity_list = np.log10(0.1+activity_list)
    enhancer_contact = np.log10(1+enhancer_contact)
    try:
        gene_mRNA_feature = mRNA_feauture.loc[gene, mRNA_feats]
    except KeyError:
        dummy_mRNA_feature = pd.DataFrame(columns=mRNA_feats)
        dummy_mRNA_feature.loc[0] = [None]*len(mRNA_feats)
        gene_mRNA_feature = dummy_mRNA_feature.loc[0]
    mRNA_promoter_feat = np.array(list(gene_mRNA_feature.values) + [promoter_signals.loc[gene, 'PromoterActivity']])
    
    if rna_method is not None:
        return pe_code, enhancer_distance, enhancer_activity, enhancer_contact, mRNA_promoter_feat, rna_signal_list
    
    return pe_code, enhancer_distance, enhancer_activity, enhancer_contact, mRNA_promoter_feat




if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)

    
    parser = argparse.ArgumentParser()

    parser.add_argument('--cell', type=str, help ='cell line (support K562 and GM12878)', choices=['K562', 'GM12878'], required=True)
    parser.add_argument('--rna', help='option for rna encoding, embedding, or one-hot incorporation', choices=['encoding', 'embedding', 'one-hot', None], default=None, required=True)

    args = parser.parse_args()

    cell = args.cell
    rna_method = args.rna

    if cell == 'K562':
        enhancer_gene_k562_100kb = pd.read_csv('./data/K562_enhancer_gene_links_100kb.hg38.tsv', sep='\t')
        promoter_signals = pd.read_csv('./data/ABC-multiTSS_nominated/K562/Neighborhoods/GeneList.txt', sep='\t')[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
        promoter_signals['ENSID'] = promoter_signals['Ensembl_ID']
        gene_enhancer_table = enhancer_gene_k562_100kb.merge(promoter_signals, left_on='TargetGeneEnsembl_ID', right_on='Ensembl_ID', how='right', suffixes=['', '_gene']).reset_index()

        if rna_method == 'encoding' or rna_method == 'one_hot':
            samples = glob.glob('./data/RNASeq_bw/K562*stranded*coverage.txt')
            rna_df_list = []
            for file in samples:
                rna_df_list.append(pd.read_csv(file, header=None, sep='\t'))
            #rna_df = pd.read_csv('./data/RNASeq_bw/K562.stranded.ENCFF336COA.ENCFF829PNJ.coverage.txt', header=None, sep='\t')
            #rna_df = pd.read_csv('./data/RNASeq_bw/K562.unstranded.ENCFF448XCV.coverage.txt', header=None, sep='\t')
            file_path = '/scratch/han_lab/dwito/EPInformer/K562_enhancer_promoter_encoding.rna_encoding.hg38.h5'

        elif rna_method == 'embedding':
            rna_df = pd.read_csv('./data/RNASeq_bw/gene_added_K562.stranded.ENCFF336COA.ENCFF829PNJ_values_TSS.tab', sep='\t', skiprows=3, header=None, index_col=0)
            file_path = '/scratch/han_lab/dwito/EPInformer/K562_enhancer_promoter_encoding.rna_embedding.hg38.h5'

    elif cell == 'GM12878':
        enhancer_gene_gm12878_100kb = pd.read_csv('./data/GM12878_enhancer_gene_links_100kb.hg38.tsv', sep='\t')
        promoter_signals = pd.read_csv('./data/ABC-multiTSS_nominated/GM12878/Neighborhoods/GeneList.txt', sep='\t')[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
        promoter_signals['ENSID'] = promoter_signals['Ensembl_ID']
        gene_enhancer_table = enhancer_gene_gm12878_100kb.merge(promoter_signals, left_on='TargetGeneEnsembl_ID', right_on='Ensembl_ID', how='right', suffixes=['', '_gene']).reset_index()

        if rna_method == 'encoding' or rna_method == 'one_hot':
            samples = glob.glob('./data/RNASeq_bw/GM12878*stranded*coverage.txt')
            rna_df_list = []
            for file in samples:
                rna_df_list.append(pd.read_csv(file, header=None, sep='\t'))
            #rna_df = pd.read_csv('./data/RNASeq_bw/GM12878.stranded.ENCFF164VLA.ENCFF074SXQ.coverage.txt', header=None, sep='\t')
            #rna_df = pd.read_csv('./data/RNASeq_bw/GM12878.unstranded.ENCFF104OTO.coverage.txt', header=None, sep='\t')
            file_path = '/scratch/han_lab/dwito/EPInformer/GM12878_enhancer_promoter_encoding.rna_encoding.hg38.h5'	

        elif rna_method == 'embedding':
            rna_df = pd.read_csv('./data/RNASeq_bw/gene_added_GM12878.stranded.ENCFF164VLA.ENCFF074SXQ_values_TSS.tab', sep='\t', skiprows=3, header=None, index_col=0)
            file_path = '/scratch/han_lab/dwito/EPInformer/GM12878_enhancer_promoter_encoding.rna_embedding.hg38.h5'

    gene_list = list(promoter_signals['ENSID'])

    max_seq_len = 2000
    add_flanking = False
    max_n_enhancer = 60
    max_distanceToTSS = 100_000
    num_features = 3
    
    mRNA_feauture = pd.read_csv('./data/RNA_CAGE.txt', sep='\t', index_col='ENSID')
    promoter_signals['PromoterActivity'] = np.sqrt(promoter_signals['H3K27ac.RPM.TSS1Kb']*promoter_signals['DHS.RPM.TSS1Kb'])
    promoter_signals.set_index('ENSID', inplace=True)
    mRNA_feats = ['UTR5LEN_log10zscore',
       'CDSLEN_log10zscore', 'INTRONLEN_log10zscore', 'UTR3LEN_log10zscore',
       'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
    PE_code_list = []
    #PE_feat_list = []
    PE_distance_list = []
    PE_activity_list = []
    PE_contact_list = []
    mRNA_promoter_list = []
    PE_links_list = []
    rna_signal_list = []
    results_list = []
    pool = Pool(processes=80)
    for gene in tqdm(pool.imap(process_gene, gene_list), total=len(gene_list)):
        if rna_method is not None:
            #pe_code, distance_list, activity_list, contact_list, mRNA_promoter_feat, gene_rna_df = gene
            results_list.append(gene)

    pool.close()
    pool.join()

    for result in results_list:
        PE_code_list.append(result[0])
        #PE_feat_list.append(PE_feat)
        PE_distance_list.append(result[1])
        PE_activity_list.append(result[2])
        PE_contact_list.append(result[3])
        mRNA_promoter_list.append(result[4])
        if rna_method is not None:
            rna_signal_list.append(result[5])

    del results_list
    del rna_df_list

    PE_code_list = np.array(PE_code_list)
    #PE_feat_list = np.array(PE_feat_list)
    PE_distance_list = np.array(PE_distance_list)
    PE_activity_list = np.array(PE_activity_list)
    PE_contact_list = np.array(PE_contact_list)
    mRNA_promoter_list = np.array(mRNA_promoter_list)
    if rna_method is not None:
        rna_signal_list = np.array(rna_signal_list)
        create_h5_data(file_path, gene_list, PE_code_list, PE_distance_list, PE_activity_list, PE_contact_list, rna_signal_list)
