#TODO: PMI calculation, add word-word file
#TODO: tfidf calculation, arrange word-sequence file

import pandas as pd
import numpy as np
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import ByteLevelBPETokenizer


def arrange_by_tfidf():

    # Read sequence word counts, calculate tfidf
    # Create file with only first 250, 500, 1000 appeared words.

    # Seq id: 0 to 2740
    # Word id: 2741 to 7501

    # Sequence-word count:
    protein_word_counts = pd.read_csv(
        'created_tables/setup5/seq_to_word_counts.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )
    protein_word_counts.drop(columns=protein_word_counts.columns[0], axis=1, inplace=True)
    protein_word_counts = protein_word_counts.iloc[1:, :]
    print(protein_word_counts)

    index = []
    for i in range(2741, 7502):
        index.append(i)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(protein_word_counts)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=index, columns=["idf_weights"])
    df_idf.sort_values(by=['idf_weights'])

    tf_idf_vector = tfidf_transformer.transform(protein_word_counts)
    feature_names = index

    first_document_vector = tf_idf_vector[1]
    df_tfifd = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    df_tfifd.sort_values(by=["tfidf"], ascending=False, inplace=True)
    print("tf-idf:", df_tfifd)

    word_ids = df_tfifd.index
    word_ids = word_ids[20:750]
    print(word_ids)

    # Now remove small tfidf value words from seq_to_word
    # Save new seq_to_word as seq_to_word_filtered
    protein_word_content = pd.read_csv(
        'created_tables/setup5/seq_to_word.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    protein_word_content.drop(columns=protein_word_content.columns[0], axis=1, inplace=True)
    protein_word_content = protein_word_content.iloc[1:, :]
    protein_word_content.columns = ['source', 'target']
    protein_word_content = protein_word_content.astype(int)
    print('Protein word content: (edges)', protein_word_content)

    delete_row_indexes = []
    for row_index, row in protein_word_content.iterrows():
        word = row['target']
        if word not in word_ids:
            delete_row_indexes.append(row_index)

    protein_word_content.drop(index=delete_row_indexes, inplace = True)
    print(protein_word_content)
    pd.DataFrame(protein_word_content).to_csv('created_tables/setup5/seq_to_word_filtered.csv', sep='\t')

def calculate_pmis():

    # ID mapping: 0 -> 4760
    #             2741 -> 7501
    # Add 2740 to get the word id

    # Get word counts
    word_counts = pd.read_csv(
        'created_tables/setup5/all_word_counts.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["index", "count"]
    )
    word_counts.drop(columns=word_counts.columns[0], axis=1, inplace=True)
    word_counts = word_counts.iloc[1:, :]
    print(word_counts)
    # Get total word count: 5555717.0
    total_word_count = word_counts['count'].sum()
    print('Total word count is:', total_word_count)
    word_counts = word_counts.to_numpy()
    print('numpy:', word_counts)
    print(word_counts[80][0])
    print(word_counts[20][0])

    # Get word co-occurance
    word_cooccurance = pd.read_csv(
        'created_tables/setup5/word_co_occurance_counts.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row

    )
    word_cooccurance.drop(columns=word_cooccurance.columns[0], axis=1, inplace=True)
    word_cooccurance = word_cooccurance.iloc[1:, :]
    print(word_cooccurance)
    word_cooccurance = word_cooccurance.to_numpy()
    print('numpy:', word_cooccurance)
    print(word_cooccurance[0][0])
    print(word_cooccurance[20][500])


    # Calculation of PMI:
    # PMI(a, b) = log2(( co-oc of a,b/total word count) / ((occurance of a/ total word count) * (occurance of b/ total word count)))

    # Calculate PMI for all word pairs:
    word_to_word_PMI = np.zeros((len(word_counts), len(word_counts)))
    all_PMI_values = []

    for i in range (4761):
        if i %10 == 0:
            print(i)
        for y in range(4761):
            # i is index of word1
            # y is index of word2
            co_oc = word_cooccurance[i][y]
            if co_oc == 0:
                continue
            else:
                nom = co_oc/total_word_count
                occ_a = word_counts[i][0]
                occ_b = word_counts[y][0]
                deno = (occ_a/total_word_count)*(occ_b/total_word_count)
                pmi = math.log((nom/deno), 2)
                word_to_word_PMI[i][y] = pmi
                all_PMI_values.append(pmi)

    print(word_to_word_PMI)
    pd.DataFrame(word_to_word_PMI).to_csv('created_tables/setup5/word_to_word_PMI.csv', sep='\t')

    all_PMI_values.sort(reverse=True)
    print(all_PMI_values[0:500])

    # Create a word-word edge file based on PMI filtering
    # Add edge if PMI>x

def main():
    print('main')
    #arrange_by_tfidf()
    calculate_pmis()

if __name__ == '__main__':
    main()