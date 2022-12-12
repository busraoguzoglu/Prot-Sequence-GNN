from collections import OrderedDict
from tokenizers import ByteLevelBPETokenizer
import pickle
import math
import pandas as pd
import numpy as np
#import nltk
#nltk.download('omw-1.4')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from Levenshtein import distance

def train_tokenizer():
    # Initialize an empty tokenizer
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    # And then train
    tokenizer.train(
        'denovo/uniprot_sprot_504k_seq.txt',
        vocab_size=5000,
        min_frequency=2,
        show_progress=True,
        special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'],
    )
    return tokenizer

def read_datasets():
    df = pd.read_pickle('denovo/train.pkl')
    df2 = pd.read_pickle('denovo/train_obsoletes.pkl')
    frames = [df, df2]
    train_frames = pd.concat(frames)

    df = pd.read_pickle('denovo/test.pkl')
    df2 = pd.read_pickle('denovo/test_obsoletes.pkl')
    frames = [df, df2]
    test_frames = pd.concat(frames)

    return train_frames, test_frames

def seq_to_id_with_types(train_frames, test_frames):

    train_sequences = []
    test_sequences = []

    for index, row in train_frames.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        train_sequences.append(sequence1)
        train_sequences.append(sequence2)

    for index, row in test_frames.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        test_sequences.append(sequence1)
        test_sequences.append(sequence2)

    unique_sequences_train = list(set(train_sequences))
    unique_sequences_test = list(set(test_sequences))

    # Assigning ids to values
    list_ids_train = [{v: k for k, v in enumerate(
        OrderedDict.fromkeys(unique_sequences_train))}
                [n] for n in unique_sequences_train]
    list_ids_test = [{v: k for k, v in enumerate(
        OrderedDict.fromkeys(unique_sequences_test))}
                [n] for n in unique_sequences_test]

    count = 0
    types_train = []
    types_test = []
    print('type finding start')
    for sequence in unique_sequences_train:
        if count % 10 == 0:
            print(count)
        type = ''
        for index, row in train_frames.iterrows():
            sequence1 = row['protein_sequence1']
            sequence2 = row['protein_sequence2']
            if sequence == sequence1:
                type = 'human'
            if sequence == sequence2:
                type = 'virus'
        types_train.append(type)
        count += 1

    for sequence in unique_sequences_test:
        if count % 10 == 0:
            print(count)
        type = ''
        for index, row in test_frames.iterrows():
            sequence1 = row['protein_sequence1']
            sequence2 = row['protein_sequence2']
            if sequence == sequence1:
                type = 'human'
            if sequence == sequence2:
                type = 'virus'
        types_test.append(type)
        count += 1

    print('type finding end')

    # The result
    dt = {'id': list_ids_train, 'sequence': unique_sequences_train, 'type': types_train}
    dt = pd.DataFrame(dt)
    print('seq ids and types, train:', dt)

    dtest = {'id': list_ids_test, 'sequence': unique_sequences_test, 'type': types_test}
    dtest = pd.DataFrame(dtest)
    print('seq ids and types, test:', dtest)

    # All id list:
    d2t = {'id': list_ids_train}
    pd.DataFrame(d2t).to_csv('created_tables/all_seq_id_train.csv', sep='\t')

    d2test = {'id': list_ids_test}
    pd.DataFrame(d2test).to_csv('created_tables/all_seq_id_test.csv', sep='\t')

    # All human id list:
    d_humt = dt.loc[dt['type'] == 'human']
    d_humt = d_humt.drop(['sequence', 'type'], axis=1)
    pd.DataFrame(d_humt).to_csv('created_tables/all_seq_id_human_train.csv', sep='\t')

    d_humtest = dtest.loc[dtest['type'] == 'human']
    d_humtest = d_humtest.drop(['sequence', 'type'], axis=1)
    pd.DataFrame(d_humtest).to_csv('created_tables/all_seq_id_human_test.csv', sep='\t')

    # All virus id list:
    d_virt = dt.loc[dt['type'] == 'virus']
    d_virt = d_virt.drop(['sequence', 'type'], axis=1)
    pd.DataFrame(d_virt).to_csv('created_tables/all_seq_id_virus_train.csv', sep='\t')

    d_virtest = dtest.loc[dtest['type'] == 'virus']
    d_virtest = d_virtest.drop(['sequence', 'type'], axis=1)
    pd.DataFrame(d_virtest).to_csv('created_tables/all_seq_id_virus_test.csv', sep='\t')

    # Do same stuff without test as well, calculate others based on that one.

    return dt, dtest

def word_to_id(seq_id_map_train, seq_id_map_test, tokenizer, train_frames):

    # Train vocab:
    vocab_train = []
    for index, row in seq_id_map_train.iterrows():
        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        for token in encoded.tokens:
            vocab_train.append(token)

    unique_words_train = list(set(vocab_train))

    # Test vocab:
    vocab_test = []
    for index, row in seq_id_map_test.iterrows():
        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        for token in encoded.tokens:
            vocab_test.append(token)

    unique_words_test = list(set(vocab_test))

    # Assigning ids to values for train
    list_ids_train = [{v: k for k, v in enumerate(
        OrderedDict.fromkeys(unique_words_train))}
                      [n] for n in unique_words_train]

    # Assigning ids to values for test
    list_ids_test = [{v: k for k, v in enumerate(
        OrderedDict.fromkeys(unique_words_test))}
                      [n] for n in unique_words_test]


    # Make ids unique for words and sequences
    list_ids_train = [x + len(seq_id_map_train) for x in list_ids_train]
    list_ids_test = [x + len(seq_id_map_test) for x in list_ids_test]

    # The result
    dt = {'id': list_ids_train, 'word': unique_words_train}
    dft = pd.DataFrame(dt)
    dtest = {'id': list_ids_test, 'word': unique_words_test}
    dftest = pd.DataFrame(dtest)

    # All id list:
    d2t = {'id': list_ids_train}
    pd.DataFrame(d2t).to_csv('created_tables/all_word_id_train.csv', sep='\t')
    d2test = {'id': list_ids_test}
    pd.DataFrame(d2test).to_csv('created_tables/all_word_id_test.csv', sep='\t')

    #######################
    #####PMI###############
    #######################
    word_counts = np.zeros((len(dft)))
    print(len(dft))
    word_to_word_counts = np.zeros((len(dft), len(dft)))

    seq_id_len = len(seq_id_map_train)

    for index, row in train_frames.iterrows():
        if index % 10 == 0:
            print('word count1:', index)

        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']

        encoded1 = tokenizer.encode(sequence1)
        encoded1 = encoded1.tokens
        encoded2 = tokenizer.encode(sequence2)
        encoded2 = encoded2.tokens

        # for token in encoded1:
        for y in range(len(encoded1)):
            # find the id of this token in df:
            row = dft.loc[dft['word'] == encoded1[y]]
            id = row['id'].values[0]
            # add one to corresponding count
            word_counts[id - seq_id_len] += 1

            # Check word co-occurances:
            for z in range(20):
                if y + z + 1 < len(encoded1):
                    token_next = encoded1[y + z + 1]
                    # get the id of token_next
                    row = dft.loc[dft['word'] == token_next]
                    id2 = row['id'].values[0]
                    # add the co-occurance to the matrix:
                    word_to_word_counts[id - seq_id_len][id2 - seq_id_len] += 1

        # for token in encoded2:
        for y in range(len(encoded2)):
            # find the id of this token in df:
            row = dft.loc[dft['word'] == encoded2[y]]
            id = row['id'].values[0]
            # add one to corresponding count
            word_counts[id - seq_id_len] += 1

            # Check word co-occurances:
            for z in range(20):
                if y + z + 1 < len(encoded2):
                    token_next = encoded2[y + z + 1]
                    # get the id of token_next
                    row = dft.loc[dft['word'] == token_next]
                    id2 = row['id'].values[0]
                    # add the co-occurance to the matrix:
                    word_to_word_counts[id - seq_id_len][id2 - seq_id_len] += 1

    # Write counts to file:
    pd.DataFrame(word_counts).to_csv('created_tables/all_word_counts.csv', sep='\t')
    pd.DataFrame(word_to_word_counts).to_csv('created_tables/word_co_occurance_counts.csv', sep='\t')
    return dft, dftest

def seq_to_word(seq_id_map_train, seq_id_map_test, word_id_map_train, word_id_map_test, tokenizer):

    # Enough to do it for train
    seq_to_word_frame_train = pd.DataFrame(columns=['source', 'target'])

    count = 0
    for row_index, row in seq_id_map_train.iterrows():

        if count % 10 == 0:
            print(count)

        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        encoded = encoded.tokens

        # Find index of tokens (words)
        for col_index, row in word_id_map_train.iterrows():

            word = row['word']
            # if word exists in encoded append edge list
            if word in encoded:
                # Instead of col_index need to use id of that word
                # Used col_index+len(seq_id_map) but if problem do smt else:
                seq_to_word_frame_train = seq_to_word_frame_train.append(
                    {'source': row_index, 'target': col_index + len(seq_id_map_train)},
                    ignore_index=True)

        count += 1


    print('seq_to_word_frame_train:', seq_to_word_frame_train)

    # Write to file
    seq_id_map_train.to_csv('created_tables/seq_id_map_train.csv', sep='\t')
    seq_id_map_test.to_csv('created_tables/seq_id_map_test.csv', sep='\t')
    word_id_map_train.to_csv('created_tables/word_id_map_train.csv', sep='\t')
    word_id_map_test.to_csv('created_tables/word_id_map_test.csv', sep='\t')
    seq_to_word_frame_train.to_csv('created_tables/seq_to_word_train.csv', sep='\t')

    # Create tfidf file:
    seq_to_word_counts = np.zeros((len(seq_id_map_train), len(word_id_map_train)))
    # Remember that word id map starts with id: 2741
    # In matrix, 0 for word will correspond to id 2741

    for row_index, row in seq_id_map_train.iterrows():
        if count % 10 == 0:
            print(count)

        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        encoded = encoded.tokens

        # Find index of tokens (words)
        for col_index, row in word_id_map_train.iterrows():
            word = row['word']
            # if word exists in encoded [index1][index2] = 1
            if word in encoded:
                # Count how many times it appears:
                count = encoded.count(word)
                seq_to_word_counts[int(row_index)][int(col_index)] += count
                if seq_to_word_counts[int(row_index)][int(col_index)] > 1:
                    print('bigger than 1')

        count += 1

    print(seq_to_word_counts)
    pd.DataFrame(seq_to_word_counts).to_csv('created_tables/seq_to_word_counts.csv', sep='\t')
    return seq_to_word_frame_train

def seq_to_seq(seq_id_map_train, seq_id_map_test, train_frames, test_frames):
    train_frames_with_labels_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2', 'interaction'])

    for table_index, row in train_frames.iterrows():
        human = row['protein_sequence1']
        virus = row['protein_sequence2']
        hum_id = 0
        vir_id = 0
        interaction = row['interaction']
        # Change sequence with sequence id
        for row_index, row in seq_id_map_train.iterrows():
            if row['sequence'] == human:
                hum_id = row['id']
            if row['sequence'] == virus:
                vir_id = row['id']
        train_frames_with_labels_new = train_frames_with_labels_new.append(
            {'protein_sequence1': hum_id, 'protein_sequence2': vir_id, 'interaction': interaction},
            ignore_index=True)

    pd.DataFrame(train_frames_with_labels_new).to_csv('created_tables/seq_to_seq_train_withlabels.csv', sep='\t')

    test_frames_with_labels_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2', 'interaction'])

    for table_index, row in test_frames.iterrows():
        human = row['protein_sequence1']
        virus = row['protein_sequence2']
        hum_id = 0
        vir_id = 0
        interaction = row['interaction']
        # Change sequence with sequence id
        for row_index, row in seq_id_map_test.iterrows():
            if row['sequence'] == human:
                hum_id = row['id']
            if row['sequence'] == virus:
                vir_id = row['id']
        test_frames_with_labels_new = test_frames_with_labels_new.append(
            {'protein_sequence1': hum_id, 'protein_sequence2': vir_id, 'interaction': interaction},
            ignore_index=True)

    pd.DataFrame(test_frames_with_labels_new).to_csv('created_tables/seq_to_seq_test_withlabels.csv', sep='\t')

def get_hum_virus_sequences():

    seq_id_map_train = pd.read_csv(
        'created_tables/clean_metapath/seq_id_map_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id", "sequence", "type"]
    )

    seq_id_map_train = seq_id_map_train.iloc[1:, :]
    print(seq_id_map_train)

    all_seq_id_human_train = pd.read_csv(
        'created_tables/clean_metapath/all_seq_id_human_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id"]
    )

    all_seq_id_human_train = all_seq_id_human_train.iloc[1:, :]
    print(all_seq_id_human_train)

    all_seq_id_virus_train = pd.read_csv(
        'created_tables/clean_metapath/all_seq_id_virus_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id"]
    )

    all_seq_id_virus_train = all_seq_id_virus_train.iloc[1:, :]
    print(all_seq_id_virus_train)

    # Add sequences to all seq id hum and virus
    hum_sequences_train = []
    vir_sequences_train = []

    for index, row in all_seq_id_human_train.iterrows():
        if index %10 == 0:
            print('word count1:', index)
        id1 = row['id']
        for index2, row2 in seq_id_map_train.iterrows():
            id2 = row2['id']
            sequence = row2['sequence']
            if id1 == id2:
                hum_sequences_train.append(sequence)

    for index, row in all_seq_id_virus_train.iterrows():
        if index %10 == 0:
            print('word count1:', index)
        id1 = row['id']
        for index2, row2 in seq_id_map_train.iterrows():
            id2 = row2['id']
            sequence = row2['sequence']
            if id1 == id2:
                vir_sequences_train.append(sequence)

    # Add new rows:
    all_seq_id_human_train['sequence'] = hum_sequences_train
    all_seq_id_virus_train['sequence'] = vir_sequences_train

    print(all_seq_id_human_train)
    print(all_seq_id_virus_train)

    all_seq_id_human_train.to_csv('created_tables/clean_metapath/all_seq_id_and_seq_human_train.csv', sep='\t')
    all_seq_id_virus_train.to_csv('created_tables/clean_metapath/all_seq_id_and_seq_virus_train.csv', sep='\t')

    # Do same for test files:
    seq_id_map_test = pd.read_csv(
        'created_tables/clean_metapath/seq_id_map_test.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id", "sequence", "type"]
    )

    seq_id_map_test = seq_id_map_test.iloc[1:, :]
    print(seq_id_map_test)

    all_seq_id_human_test = pd.read_csv(
        'created_tables/clean_metapath/all_seq_id_human_test.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id"]
    )

    all_seq_id_human_test = all_seq_id_human_test.iloc[1:, :]
    print(all_seq_id_human_test)

    all_seq_id_virus_test = pd.read_csv(
        'created_tables/clean_metapath/all_seq_id_virus_test.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id"]
    )

    all_seq_id_virus_test = all_seq_id_virus_test.iloc[1:, :]
    print(all_seq_id_virus_test)

    # Add sequences to all seq id hum and virus
    hum_sequences_test = []
    vir_sequences_test = []

    for index, row in all_seq_id_human_test.iterrows():
        if index % 10 == 0:
            print('word count1:', index)
        id1 = row['id']
        for index2, row2 in seq_id_map_test.iterrows():
            id2 = row2['id']
            sequence = row2['sequence']
            if id1 == id2:
                hum_sequences_test.append(sequence)

    for index, row in all_seq_id_virus_test.iterrows():
        if index % 10 == 0:
            print('word count1:', index)
        id1 = row['id']
        for index2, row2 in seq_id_map_test.iterrows():
            id2 = row2['id']
            sequence = row2['sequence']
            if id1 == id2:
                vir_sequences_test.append(sequence)

    # Add new rows:
    all_seq_id_human_test['sequence'] = hum_sequences_test
    all_seq_id_virus_test['sequence'] = vir_sequences_test

    print(all_seq_id_human_test)
    print(all_seq_id_virus_test)

    all_seq_id_human_test.to_csv('created_tables/clean_metapath/all_seq_id_and_seq_human_test.csv', sep='\t')
    all_seq_id_virus_test.to_csv('created_tables/clean_metapath/all_seq_id_and_seq_virus_test.csv', sep='\t')

def arrange_by_tfidf():

    # Read sequence word counts, calculate tfidf
    # Create file with only first 250, 500, 1000 appeared words.

    # Seq id train: 0 to 2647
    # Word id train: 2648 to 7402

    # Sequence-word count:
    protein_word_counts = pd.read_csv(
        'created_tables/clean_metapath/seq_to_word_counts.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )
    protein_word_counts.drop(columns=protein_word_counts.columns[0], axis=1, inplace=True)
    protein_word_counts = protein_word_counts.iloc[1:, :]
    print(protein_word_counts)

    index = []
    for i in range(2648, 7403):
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
    word_ids = word_ids[0:250]
    print(word_ids)

    # Now remove small tfidf value words from seq_to_word
    # Save new seq_to_word as seq_to_word_filtered
    protein_word_content = pd.read_csv(
        'created_tables/clean_metapath/seq_to_word_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    protein_word_content.drop(columns=protein_word_content.columns[0], axis=1, inplace=True)
    protein_word_content = protein_word_content.iloc[1:, :]
    protein_word_content.columns = ['source', 'target']
    print('Protein word content: (edges)', protein_word_content)

    delete_row_indexes = []
    for row_index, row in protein_word_content.iterrows():
        word = row['target']
        if int(float(word)) not in word_ids:
            delete_row_indexes.append(row_index)

    protein_word_content.drop(index=delete_row_indexes, inplace = True)
    print(protein_word_content)
    pd.DataFrame(protein_word_content).to_csv('created_tables/clean_metapath/seq_to_word_train_tfidf_filtered.csv', sep='\t')

def calculate_pmis():

    # ID mapping: 0 -> 4760
    #             2741 -> 7501

    # ID mapping: 0 -> 4754
    #             2648 -> 7402

    # Get word counts
    word_counts = pd.read_csv(
        'created_tables/clean_metapath/all_word_counts.csv',
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
        'created_tables/clean_metapath/word_co_occurance_counts.csv',
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

    for i in range (4755):
        if i %10 == 0:
            print(i)
        for y in range(4755):
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
    pd.DataFrame(word_to_word_PMI).to_csv('created_tables/clean_metapath/word_to_word_PMI.csv', sep='\t')

    all_PMI_values.sort(reverse=True)
    print(all_PMI_values[0:500])

    # Create a word-word edge file based on PMI filtering
    # Add edge if PMI>x
    word_word_edges = pd.DataFrame(columns=['source', 'target'])
    source = []
    target = []
    pmi_cutoff = 13

    for i in range(4755):
        for y in range(4755):
            pmi = word_to_word_PMI[i][y]
            if pmi > pmi_cutoff:
                source.append(i+2648)
                target.append(y+2648)

    word_word_edges['source'] = source
    word_word_edges['target'] = target
    word_word_edges.to_csv('created_tables/clean_metapath/word_to_word_edges.csv', sep='\t')

def create_similarity_edges():

    df1 = pd.read_csv('created_tables/clean_metapath/all_seq_id_and_seq_human_train.csv',
                     sep="\t",  # tab-separated
                     header=None,  # no heading row
                     names=["id", "sequence"])
    df1 = df1.iloc[1:, :]

    df2 = pd.read_csv('created_tables/clean_metapath/all_seq_id_and_seq_virus_train.csv',
                      sep="\t",  # tab-separated
                      header=None,  # no heading row
                      names=["id", "sequence"])
    df2 = df2.iloc[1:, :]

    index_train = []
    index_test = []

    for i in range(len(df1)):
        index_train.append(i)

    for i in range(len(df2)):
        index_test.append(i)

    df1['index'] = index_train
    df2['index'] = index_test

    df1.set_index('index', inplace = True)
    df2.set_index('index', inplace = True)

    print(df1)
    print(df2)

    simil_hum1 = []
    simil_hum2 = []

    simil_vir1 = []
    simil_vir2 = []

    # TODO:
    # Get test set, if sequence is in test set, do not add an edge:

    testdf = pd.read_csv('created_tables/clean_metapath/seq_to_seq_test_withlabels.csv',
                      sep="\t",  # tab-separated
                      header=None,  # no heading row
                      names=["protein_sequence1", "protein_sequence2", "interaction"])
    testdf = testdf.iloc[1:, :]

    print('test:', testdf)

    found = testdf[testdf['protein_sequence1'].str.contains('1076')]
    print('found count:', len(found))
    test_count = 0

    for i in range(len(df1)):
        id = df1.loc[i, "id"]
        sequence = df1.loc[i, "sequence"]

        # check if id exist in test:
        if id in testdf['protein_sequence1'].values:
            test_count += 1
            continue

        for y in range(i+1, len(df1)):
            id2 = df1.loc[y, "id"]
            sequence2 = df1.loc[y, "sequence"]

            if sequence != sequence2:
                lev = distance(sequence, sequence2)
                if lev < 150:
                    # For < 200 (arbitrary number) add edge
                    simil_hum1.append(id)
                    simil_hum2.append(id2)

    print('test eliminated count:', test_count)
    test_count = 0

    for i in range(len(df2)):
        id = df2.loc[i, "id"]
        sequence = df2.loc[i, "sequence"]

        # check if id exist in test:
        # check if id exist in test:
        if id in testdf['protein_sequence2'].values:
            test_count+=1
            continue

        for y in range(i+1, len(df2)):
            id2 = df2.loc[y, "id"]
            sequence2 = df2.loc[y, "sequence"]

            if sequence != sequence2:
                lev = distance(sequence, sequence2)
                if lev < 150:
                    # For < 200 (arbitrary number) add edge
                    simil_vir1.append(id)
                    simil_vir2.append(id2)

    print('test eliminated count:', test_count)

    df_hum = pd.DataFrame(list(zip(simil_hum1, simil_hum2)),
                      columns=['source', 'target'])
    print(df_hum)

    df_vir = pd.DataFrame(list(zip(simil_vir1, simil_vir2)),
                          columns=['source', 'target'])
    print(df_vir)

    df_hum.to_csv('created_tables/clean_metapath/lev_dist_edges_hum.csv', sep='\t')
    df_vir.to_csv('created_tables/clean_metapath/lev_dist_edges_vir.csv', sep='\t')

def create_similarity_edges_jaccard():

    # Get sequences:
    df1 = pd.read_csv('created_tables/clean_metapath/all_seq_id_human_train.csv',
                      sep="\t",  # tab-separated
                      header=None,  # no heading row
                      names=["id"])
    df1 = df1.iloc[1:, :]
    print(df1)
    df1 = df1.astype(int)

    df2 = pd.read_csv('created_tables/clean_metapath/all_seq_id_virus_train.csv',
                      sep="\t",  # tab-separated
                      header=None,  # no heading row
                      names=["id"])
    df2 = df2.iloc[1:, :]
    df2 = df2.astype(int)
    print(df2)

    # Get sequence-word edges:
    seqword = pd.read_csv('created_tables/clean_metapath/seq_to_word_train.csv',
                         sep="\t",  # tab-separated
                         header=None,  # no heading row
                         names=["source", "target"])
    seqword = seqword.iloc[1:, :]

    # convert all to int
    seqword = seqword.astype(float)
    seqword = seqword.astype(int)
    print(seqword)

    # TODO:
    # Get test set, if sequence is in test set, do not add an edge:

    testdf = pd.read_csv('created_tables/clean_metapath/seq_to_seq_test_withlabels.csv',
                         sep="\t",  # tab-separated
                         header=None,  # no heading row
                         names=["protein_sequence1", "protein_sequence2", "interaction"])
    testdf = testdf.iloc[1:, :]

    print('test:', testdf)

    found = testdf[testdf['protein_sequence1'].str.contains('1076')]
    print('found count:', len(found))

    # Indexes set to be able to iterate
    index_train = []
    index_test = []

    for i in range(len(df1)):
        index_train.append(i)

    for i in range(len(df2)):
        index_test.append(i)

    df1['index'] = index_train
    df2['index'] = index_test

    df1.set_index('index', inplace=True)
    df2.set_index('index', inplace=True)

    # Iterate over all human sequences
    # For each sequence, get the list of words it contains:

    # Same index on these two will contain sequence id and
    # list of word ids that sequence has
    human_sequence_id_list = []
    human_sequence_word_list = []
    count = 0

    test_count = 0

    for i in range(len(df1)):

        #if count%10 == 0 & count > 0:
            #print(count)

        id = df1.loc[i, "id"]
        id = int(id)

        # check if id exist in test:
        if id in testdf['protein_sequence1'].values:
            test_count += 1
            continue


        # Get list of words that this id contains:
        # Filter seqword by source being id
        # add all target to a list and append human sequence word list
        all_words_df = seqword[seqword['source'] == id]
        word_list = all_words_df['target'].tolist()
        human_sequence_id_list.append(id)
        human_sequence_word_list.append(word_list)
        count+=1

    print('test eliminated count:', test_count)

    virus_sequence_id_list = []
    virus_sequence_word_list = []
    count = 0

    test_count = 0

    for i in range(len(df2)):
        # if count%10 == 0 & count > 0:
        # print(count)

        id = df2.loc[i, "id"]
        id = int(id)

        # check if id exist in test:
        # check if id exist in test:
        if id in testdf['protein_sequence2'].values:
            test_count += 1
            continue

        # Get list of words that this id contains:
        # Filter seqword by source being id
        # add all target to a list and append human sequence word list
        all_words_df = seqword[seqword['source'] == id]
        word_list = all_words_df['target'].tolist()
        virus_sequence_id_list.append(id)
        virus_sequence_word_list.append(word_list)
        count += 1

    print('test eliminated count:', test_count)

    # Got lists that contain id-word list pairs

    # Need to calculate sequence-sequence jaccard similarity
    # Calculate for all seq-seq, add pair to edge list if similarity > something

    simil_hum1 = []
    simil_hum2 = []

    simil_vir1 = []
    simil_vir2 = []

    jac_cutoff = 0.4

    # For human:
    for i in range(len(human_sequence_id_list)):
        id1 = human_sequence_id_list[i]
        id1words = human_sequence_word_list[i]
        for y in range(i+1, len(human_sequence_id_list)):
            id2 = human_sequence_id_list[y]
            id2words = human_sequence_word_list[y]
            # Calculate jc:
            s1 = set(id1words)
            s2 = set(id2words)
            jaccard = float(len(s1.intersection(s2)) / len(s1.union(s2)))
            if jaccard > jac_cutoff:
                # Add the edge!
                simil_hum1.append(id1)
                simil_hum2.append(id2)

    for i in range(len(virus_sequence_id_list)):
        id1 = virus_sequence_id_list[i]
        id1words = virus_sequence_word_list[i]
        for y in range(i+1, len(virus_sequence_id_list)):
            id2 = virus_sequence_id_list[y]
            id2words = virus_sequence_word_list[y]
            # Calculate jc:
            s1 = set(id1words)
            s2 = set(id2words)
            jaccard = float(len(s1.intersection(s2)) / len(s1.union(s2)))
            if jaccard > jac_cutoff:
                # Add the edge!
                simil_vir1.append(id1)
                simil_vir2.append(id2)

    df_hum = pd.DataFrame(list(zip(simil_hum1, simil_hum2)),
                          columns=['source', 'target'])
    print(df_hum)

    df_vir = pd.DataFrame(list(zip(simil_vir1, simil_vir2)),
                          columns=['source', 'target'])
    print(df_vir)

    df_hum.to_csv('created_tables/clean_metapath/jaccard_sim_edges_hum.csv', sep='\t')
    df_vir.to_csv('created_tables/clean_metapath/jaccard_sim_edges_vir.csv', sep='\t')

def main():

    # Tokenizer train
    #tokenizer = train_tokenizer()
    # Save the tokenizer to a file
    #with open('tokenizer/tokenizer.pickle', 'wb') as handle:
    #    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_frames, test_frames = read_datasets()

    # load tokenizer
    with open('tokenizer/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print(train_frames)
    print(test_frames)

    # Unique sequences to id mapping
    #seq_id_map_train, seq_id_map_test = seq_to_id_with_types(train_frames, test_frames)

    # Unique words to id mapping
    #word_id_map_train, word_id_map_test = word_to_id(seq_id_map_train, seq_id_map_test, tokenizer, train_frames)

    # Load files:
    seq_id_map_train = pd.read_csv('created_tables/clean_metapath/seq_id_map_train.csv',
                                   sep="\t",  # tab-separated
                                   header=None,  # no heading row
                                   names=["id", "sequence", "type"])
    seq_id_map_train = seq_id_map_train.iloc[1:, :]

    seq_id_map_test = pd.read_csv('created_tables/clean_metapath/seq_id_map_test.csv',
                                   sep="\t",  # tab-separated
                                   header=None,  # no heading row
                                   names=["id", "sequence", "type"])
    seq_id_map_test = seq_id_map_test.iloc[1:, :]

    word_id_map_train = pd.read_csv('created_tables/clean_metapath/word_id_map_train.csv',
                                  sep="\t",  # tab-separated
                                  header=None,  # no heading row
                                  names=["id", "word"])
    word_id_map_train = word_id_map_train.iloc[1:, :]

    word_id_map_test = pd.read_csv('created_tables/clean_metapath/word_id_map_test.csv',
                                    sep="\t",  # tab-separated
                                    header=None,  # no heading row
                                    names=["id", "word"])
    word_id_map_test = word_id_map_test.iloc[1:, :]

    # Prot sequence id to word
    #seq_to_word_map_train = seq_to_word(seq_id_map_train, seq_id_map_test, word_id_map_train, word_id_map_test, tokenizer)

    # Prot sequence id to sequence id
    #seq_to_seq_map = seq_to_seq(seq_id_map_train, seq_id_map_test, train_frames, test_frames)

    #get_hum_virus_sequences()
    #arrange_by_tfidf()
    #calculate_pmis()
    #create_similarity_edges()
    create_similarity_edges_jaccard()

if __name__ == '__main__':
    main()