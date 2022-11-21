#TODO: PMI calculation, add word-word file
#TODO: tfidf calculation, arrange word-sequence file

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def arrange_by_tfidf():

    # Read sequence word counts, calculate tfidf
    # Create file with only first 250, 500, 1000 appeared words.

    # Seq id: 0 to 2740
    # Word id: 2741 to 7501

    # Sequence-word count:
    protein_word_counts = pd.read_csv(
        'created_tables/setup4_2/seq_to_word_counts.csv',
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
    word_ids = word_ids[0:750]
    print(word_ids)

    # Now remove small tfidf value words from seq_to_word
    # Save new seq_to_word as seq_to_word_filtered
    protein_word_content = pd.read_csv(
        'created_tables/setup4_2/seq_to_word.csv',
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
    pd.DataFrame(protein_word_content).to_csv('created_tables/setup4_2/seq_to_word_filtered.csv', sep='\t')

def main():
    print('main')
    arrange_by_tfidf()

if __name__ == '__main__':
    main()