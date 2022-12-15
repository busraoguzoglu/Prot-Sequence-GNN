import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from Bio.Emboss.Applications import NeedleCommandline
from multiprocessing import Pool


def needle_align_code(query_seq, target_seq):
    needle_cline = NeedleCommandline(asequence="asis:" + query_seq,
                                     bsequence="asis:" + target_seq,
                                     aformat="simple",
                                     gapopen=10,
                                     gapextend=0.5,
                                     outfile='stdout'
                                     )
    out_data, err = needle_cline()
    out_split = out_data.split("\n")
    p = re.compile("\((.*)\)")
    return float(p.search(out_split[25]).group(1).replace("%", "").strip())


def get_similarity(y, prot_id):
    seq2 = protein_sequences[protein_ids[y]]
    return needle_align_code(protein_sequences[prot_id], seq2), prot_id, protein_ids[y]


if __name__ == '__main__':
    #proteins = pd.read_csv('data/BindingDB_1chain_proteins_all.csv', index_col=False)
    #protein_sequences = pd.read_csv('data/BindingDB_onechain_protein_seq.csv', index_col=False)
    #proteins_w_seq = pd.merge(proteins, protein_sequences, on='UniProt_S_ID', how='inner').drop_duplicates(subset='UniProt_S_ID')
    proteins_w_seq = pd.read_csv(
        'hum_all_seq_id.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["index", "id", "sequence"]  # set our own names for the columns
    )
    proteins_w_seq = proteins_w_seq.iloc[1:, :]
    proteins_w_seq['id'] = proteins_w_seq['id'].astype(int)
    proteins_w_seq.drop(columns=proteins_w_seq.columns[0], axis=1, inplace=True)

    print(proteins_w_seq)
    n_protein = proteins_w_seq.shape[0]

    protein_ids = proteins_w_seq['id'].values.tolist()
    protein_sequences = dict(zip(proteins_w_seq.id, proteins_w_seq.sequence))
    similarities = np.zeros((n_protein, n_protein))

    similarities_df = pd.DataFrame(data=similarities, index=protein_ids, columns=protein_ids)

    pool = Pool(6)
    for x, protein_id in tqdm(enumerate(protein_ids)):
        print(protein_id)
        seq1 = protein_sequences[protein_id]
        sim_list = pool.starmap(get_similarity, [(y, protein_id) for y in range(x, len(protein_ids))])
        for sim, x, y in sim_list:
            similarities_df[x][y] = sim

    similarities_df.to_csv('protein_similarities_human.csv')
