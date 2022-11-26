from collections import OrderedDict
from tokenizers import ByteLevelBPETokenizer

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import torch

import nltk
nltk.download('omw-1.4')

#https://github.com/mheinzinger/SeqVec
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path

def train_tokenizer():
    # Initialize an empty tokenizer
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    # And then train
    tokenizer.train(
        'denovo/train_bpe.txt',
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

def seq_to_id(train_frames, test_frames):

    all_sequences = []
    for index, row in train_frames.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        all_sequences.append(sequence1)
        all_sequences.append(sequence2)

    for index, row in test_frames.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        all_sequences.append(sequence1)
        all_sequences.append(sequence2)

    unique_sequences = list(set(all_sequences))

    # Assigning ids to values
    list_ids = [{v: k for k, v in enumerate(
        OrderedDict.fromkeys(unique_sequences))}
                [n] for n in unique_sequences]

    # The result
    d = {'id': list_ids, 'sequence': unique_sequences}
    df = pd.DataFrame(d)
    print(df)

    # All id list:
    d2 = {'id': list_ids}
    pd.DataFrame(d2).to_csv('created_tables/all_seq_id.csv', sep='\t')

    return df

def seq_to_id_with_types(train_frames, test_frames):

    all_sequences = []
    for index, row in train_frames.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        all_sequences.append(sequence1)
        all_sequences.append(sequence2)

    for index, row in test_frames.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        all_sequences.append(sequence1)
        all_sequences.append(sequence2)

    unique_sequences = list(set(all_sequences))

    # Assigning ids to values
    list_ids = [{v: k for k, v in enumerate(
        OrderedDict.fromkeys(unique_sequences))}
                [n] for n in unique_sequences]

    count = 0
    types = []
    print('type finding start')
    for sequence in unique_sequences:
        if count %10 == 0:
            print(count)
        type = ''
        for index, row in train_frames.iterrows():
            sequence1 = row['protein_sequence1']
            sequence2 = row['protein_sequence2']
            if sequence == sequence1:
                type = 'human'
            if sequence == sequence2:
                type = 'virus'
        for index, row in test_frames.iterrows():
            sequence1 = row['protein_sequence1']
            sequence2 = row['protein_sequence2']
            if sequence == sequence1:
                type = 'human'
            if sequence == sequence2:
                type = 'virus'
        types.append(type)
        count +=1

    print('type finding end')

    # The result
    d = {'id': list_ids, 'sequence': unique_sequences, 'type': types}
    df = pd.DataFrame(d)
    print('seq ids and types:', df)

    # All id list:
    d2 = {'id': list_ids}
    pd.DataFrame(d2).to_csv('created_tables/all_seq_id.csv', sep='\t')

    # All human id list:
    d_hum = df.loc[df['type'] == 'human']
    d_hum = d_hum.drop(['sequence', 'type'], axis=1)
    pd.DataFrame(d_hum).to_csv('created_tables/all_seq_id_human.csv', sep='\t')

    # All virus id list:
    d_vir = df.loc[df['type'] == 'virus']
    d_vir = d_vir.drop(['sequence', 'type'], axis=1)
    pd.DataFrame(d_vir).to_csv('created_tables/all_seq_id_virus.csv', sep='\t')

    return df

def word_to_id(seq_id_map, tokenizer, train_frames, test_frames):

    vocab = []
    for index, row in seq_id_map.iterrows():
        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        for token in encoded.tokens:
            vocab.append(token)

    unique_words = list(set(vocab))

    # Assigning ids to values
    list_ids = [{v: k for k, v in enumerate(
        OrderedDict.fromkeys(unique_words))}
                [n] for n in unique_words]

    # Make ids unique for words and sequences
    list_ids = [x + len(seq_id_map) for x in list_ids]

    # The result
    d = {'id': list_ids, 'word': unique_words}
    df = pd.DataFrame(d)

    # All id list:
    d2 = {'id': list_ids}
    pd.DataFrame(d2).to_csv('created_tables/all_word_id.csv', sep='\t')

    print(df)

    #######################
    #####PMI###############
    #######################

    """
    # TODO: Need to calculate word-word co-occurance matrix and save that
    word_to_word_counts = np.zeros((4761, 4761))
    # Use train_bpe as text 'denovo/train_bpe.txt'
    with open('denovo/train_bpe.txt') as file:
        lines = [line.rstrip() for line in file]

    line_count = 0

    for line in lines:

        if line_count % 10 == 0:
            print('co-oc line:', line_count)

        # First tokenize:
        encoded = tokenizer.encode(line)
        encoded = encoded.tokens

        # Always look forward for co occurance:
        for i in range(len(encoded)):
            # Check next 20 tokens:
            token = encoded[i]
            # get id of token:
            row = df.loc[df['word'] == token]
            id1 = row['id']
            print('id1:', id1)
            for y in range(20):
                if i + y + 1 < len(encoded):
                    token_next = encoded[i + y + 1]
                    # get the id of token_next
                    row = df.loc[df['word'] == token_next]
                    id2 = row['id']
                    print('id2:', id2)
                    # add the co-occurance to the matrix:
                    word_to_word_counts[id1 - 2741][id2 - 2741] += 1

        line_count += 1

    # Write co-occurance counts to the file:
    pd.DataFrame(word_to_word_counts).to_csv('created_tables/word_co_occurance_counts.csv', sep='\t')
    # Note: In all PMI files, id of 0 for words correspond to id of 2741
    
    
    
    """


    ########### BELOW WORKING
    # TODO: Need to calculate word count in the text and save that
    word_counts = np.zeros((len(df)))
    print(len(df))
    word_to_word_counts = np.zeros((len(df), len(df)))
    print(word_to_word_counts)
    print(word_to_word_counts[2000][2000])
    # index 0 will correspond to id 2741
    # when we see that word, we will increment its index

    for index, row in train_frames.iterrows():
        if index %10 == 0:
            print('word count1:', index)

        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']

        encoded1 = tokenizer.encode(sequence1)
        encoded1 = encoded1.tokens
        encoded2 = tokenizer.encode(sequence2)
        encoded2 = encoded2.tokens

        #for token in encoded1:
        for y in range (len(encoded1)):
            # find the id of this token in df:
            row = df.loc[df['word'] == encoded1[y]]
            id = row['id'].values[0]
            # add one to corresponding count
            word_counts[id-2741] += 1

            # Check word co-occurances:
            for z in range(20):
                if y + z + 1 < len(encoded1):
                    token_next = encoded1[y + z + 1]
                    # get the id of token_next
                    row = df.loc[df['word'] == token_next]
                    id2 = row['id'].values[0]
                    # add the co-occurance to the matrix:
                    #print('id:', id-2741)
                    #print('id2:', id2-2741)
                    word_to_word_counts[id - 2741][id2 - 2741] += 1

        #for token in encoded2:
        for y in range(len(encoded2)):
            # find the id of this token in df:
            row = df.loc[df['word'] == encoded2[y]]
            id = row['id'].values[0]
            # add one to corresponding count
            word_counts[id-2741] += 1

            # Check word co-occurances:
            for z in range(20):
                if y + z + 1 < len(encoded2):
                    token_next = encoded2[y + z + 1]
                    # get the id of token_next
                    row = df.loc[df['word'] == token_next]
                    id2 = row['id'].values[0]
                    # add the co-occurance to the matrix:
                    word_to_word_counts[id - 2741][id2 - 2741] += 1

    for index, row in test_frames.iterrows():

        if index %10 == 0:
            print('word count2:', index)

        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']

        encoded1 = tokenizer.encode(sequence1)
        encoded1 = encoded1.tokens

        encoded2 = tokenizer.encode(sequence2)
        encoded2 = encoded2.tokens

        # for token in encoded1:
        for y in range(len(encoded1)):
            # find the id of this token in df:
            row = df.loc[df['word'] == encoded1[y]]
            id = row['id'].values[0]
            # add one to corresponding count
            word_counts[id - 2741] += 1

            # Check word co-occurances:
            for z in range(20):
                if y + z + 1 < len(encoded1):
                    token_next = encoded1[y + z + 1]
                    # get the id of token_next
                    row = df.loc[df['word'] == token_next]
                    id2 = row['id'].values[0]
                    # add the co-occurance to the matrix:
                    word_to_word_counts[id - 2741][id2 - 2741] += 1

        # for token in encoded2:
        for y in range(len(encoded2)):
            # find the id of this token in df:
            row = df.loc[df['word'] == encoded2[y]]
            id = row['id'].values[0]
            # add one to corresponding count
            word_counts[id - 2741] += 1

            # Check word co-occurances:
            for z in range(20):
                if y + z + 1 < len(encoded2):
                    token_next = encoded2[y + z + 1]
                    # get the id of token_next
                    row = df.loc[df['word'] == token_next]
                    id2 = row['id'].values[0]
                    # add the co-occurance to the matrix:
                    word_to_word_counts[id - 2741][id2 - 2741] += 1

    # Write counts to file:
    pd.DataFrame(word_counts).to_csv('created_tables/all_word_counts.csv', sep='\t')
    pd.DataFrame(word_to_word_counts).to_csv('created_tables/word_co_occurance_counts.csv', sep='\t')

    return df

def seq_to_word(seq_id_map, word_id_map, tokenizer):

    seq_to_word = np.zeros((len(seq_id_map), len(word_id_map)))
    print(seq_to_word.shape)
    done = 0
    count = 0

    for row_index, row in seq_id_map.iterrows():

        if count % 10 == 0:
            print(count)

        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        encoded = encoded.tokens

        # Find index of tokens (words)
        for col_index, row in word_id_map.iterrows():

            word = row['word']
            # if word exists in encoded [index1][index2] = 1
            if word in encoded:
                seq_to_word[row_index][col_index] = 1
                done += 1

        count+=1

    print(seq_to_word)
    print(seq_to_word.shape)
    print(done)

    # Write to file
    seq_id_map.to_csv('created_tables/seq_id_map.csv', sep='\t')
    word_id_map.to_csv('created_tables/word_id_map.csv', sep='\t')

    df = pd.DataFrame(seq_to_word)
    print(df)

    pd.DataFrame(seq_to_word).to_csv('created_tables/seq_to_word.csv', sep='\t')

def seq_to_word2(seq_id_map, word_id_map, tokenizer):

    print(word_id_map)
    seq_to_word_frame = pd.DataFrame(columns=['source', 'target'])

    count = 0
    for row_index, row in seq_id_map.iterrows():

        if count % 10 == 0:
            print(count)

        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        encoded = encoded.tokens

        # Find index of tokens (words)
        for col_index, row in word_id_map.iterrows():

            word = row['word']
            # if word exists in encoded append edge list
            if word in encoded:
                # Instead of col_index need to use id of that word
                # Used col_index+len(seq_id_map) but if problem do smt else:
                seq_to_word_frame = seq_to_word_frame.append({'source': row_index, 'target': col_index+len(seq_id_map)},
                                                           ignore_index=True)

        count+=1


    # Write to file
    seq_id_map.to_csv('created_tables/seq_id_map.csv', sep='\t')
    word_id_map.to_csv('created_tables/word_id_map.csv', sep='\t')
    seq_to_word_frame.to_csv('created_tables/seq_to_word.csv', sep='\t')

    # Create tfidf file:

    seq_to_word_counts = np.zeros((len(seq_id_map), len(word_id_map)))
    # Remember that word id map starts with id: 2741
    # In matrix, 0 for word will correspond to id 2741

    for row_index, row in seq_id_map.iterrows():
        if count % 10 == 0:
            print(count)

        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        encoded = encoded.tokens

        # Find index of tokens (words)
        for col_index, row in word_id_map.iterrows():
            word = row['word']
            # if word exists in encoded [index1][index2] = 1
            if word in encoded:
                # Count how many times it appears:
                count = encoded.count(word)
                seq_to_word_counts[row_index][col_index] += count
                if seq_to_word_counts[row_index][col_index] >1:
                    print('bigger than 1')

        count += 1

    print(seq_to_word_counts)
    pd.DataFrame(seq_to_word_counts).to_csv('created_tables/seq_to_word_counts.csv', sep='\t')

    return seq_to_word_frame

def seq_to_word_tfidf(seq_id_map, word_id_map, tokenizer):

    seq_to_word = np.zeros((len(seq_id_map), len(word_id_map)))
    print(seq_to_word.shape)
    done = 0
    count = 0

    print(seq_id_map)

    for row_index, row in seq_id_map.iterrows():
        if count % 10 == 0:
            print(count)

        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        encoded = encoded.tokens

        # Find index of tokens (words)
        for col_index, row in word_id_map.iterrows():

            word = row['word']
            # if word exists in encoded [index1][index2] = 1
            if word in encoded:
                print('increment')
                seq_to_word[row_index][col_index] += 1
                done += 1

        count+=1

    print('seq_to_word counts:', seq_to_word)
    print(seq_to_word.shape)
    print(done)

    index = []
    for i in range(len(word_id_map)):
        index.append(i)


    # seq_to_word now shows word counts
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(seq_to_word)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=index, columns=["idf_weights"])
    df_idf.sort_values(by=['idf_weights'])

    tf_idf_vector = tfidf_transformer.transform(seq_to_word)
    feature_names = index

    first_document_vector = tf_idf_vector[1]
    df_tfifd = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    df_tfifd.sort_values(by=["tfidf"], ascending=False)
    print("tf-idf:", df_tfifd)

    print(df_tfifd.iloc[4])
    print(df_tfifd.iloc[4]["tfidf"])

    # Rework seq to word:
    df_tfifd = df_tfifd[0:500]
    print('df_tfifd first 500:', df_tfifd)

    seq_to_word_tfidf = np.zeros((len(seq_id_map), len(df_tfifd)))

    for row_index, row in seq_id_map.iterrows():

        if count % 10 == 0:
            print(count)

        sequence = row['sequence']
        encoded = tokenizer.encode(sequence)
        encoded = encoded.tokens

        # Find index of tokens (words)
        for col_index, row in df_tfifd.iterrows():

            word = row['word']
            # if word exists in encoded [index1][index2] = 1
            if word in encoded:
                a = df_tfifd.iloc[col_index]
                seq_to_word_tfidf[row_index][col_index] = a["tfidf"]
                done += 1

        count += 1

    print(seq_to_word_tfidf)
    print(seq_to_word_tfidf.shape)
    print(done)

    # Write to file
    seq_id_map.to_csv('created_tables/seq_id_map.csv', sep='\t')
    word_id_map.to_csv('created_tables/word_id_map.csv', sep='\t')
    df_tfifd.to_csv('created_tables/word_tfidf.csv', sep='\t')

    pd.DataFrame(seq_to_word).to_csv('created_tables/seq_to_word.csv', sep='\t')
    pd.DataFrame(seq_to_word_tfidf).to_csv('created_tables/seq_to_word_tfidfvalues.csv', sep='\t')

def seq_to_seq(seq_id_map, train_frames, test_frames):

    seq_to_seq = np.zeros((len(seq_id_map), len(seq_id_map)))

    frames = [train_frames, test_frames]
    all_data = pd.concat(frames)

    count = 0

    for row_index, row in seq_id_map.iterrows():

        print(count)
        if count % 10 == 0:
            print(count)

        row_sequence = row['sequence']

        row_human_flag = 0
        for table_index, row in all_data.iterrows():
            human = row['protein_sequence1']
            if row_sequence == human:
                row_human_flag = 1
                break

        if row_human_flag:
            for col_index, row in seq_id_map.iterrows():
                col_sequence = row['sequence']
                for table_index, row in all_data.iterrows():
                    virus = row['protein_sequence2']

                    if virus == col_sequence:
                        interaction = row['interaction']
                        seq_to_seq[row_index][col_index] = interaction
                        seq_to_seq[col_index][row_index] = interaction

        count +=1

    pd.DataFrame(seq_to_seq).to_csv('created_tables/seq_to_seq.csv', sep='\t')

def seq_to_seq2(seq_id_map, train_frames, test_frames):
    seq_to_seq = np.zeros((len(seq_id_map), len(seq_id_map)))

    frames = [train_frames, test_frames]
    all_data = pd.concat(frames)

    count = 0

    for table_index, row in all_data.iterrows():
        human = row['protein_sequence1']
        virus = row['protein_sequence2']

        hum_index = 0
        vir_index = 0

        for row_index, row in seq_id_map.iterrows():

            if count%10 == 0:
                print(count)

            row_sequence = row['sequence']

            if row_sequence == human:
                hum_index = row_index
            if row_sequence == virus:
                vir_index = row_index

        seq_to_seq[hum_index][vir_index] = 1
        seq_to_seq[vir_index][hum_index] = 1

        count += 1

    pd.DataFrame(seq_to_seq).to_csv('created_tables/seq_to_seq.csv', sep='\t')

def seq_to_seq3(seq_id_map, train_frames, test_frames):

    train_frames_with_labels_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2', 'interaction'])

    for table_index, row in train_frames.iterrows():
        human = row['protein_sequence1']
        virus = row['protein_sequence2']
        hum_id = 0
        vir_id = 0
        interaction = row['interaction']
        # Change sequence with sequence id
        for row_index, row in seq_id_map.iterrows():
            if row['sequence'] == human:
                hum_id = row['id']
            if row['sequence'] == virus:
                vir_id = row['id']
        train_frames_with_labels_new = train_frames_with_labels_new.append({'protein_sequence1': hum_id, 'protein_sequence2': vir_id, 'interaction': interaction},
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
        for row_index, row in seq_id_map.iterrows():
            if row['sequence'] == human:
                hum_id = row['id']
            if row['sequence'] == virus:
                vir_id = row['id']
        test_frames_with_labels_new = test_frames_with_labels_new.append({'protein_sequence1': hum_id, 'protein_sequence2': vir_id, 'interaction': interaction},
                                                   ignore_index=True)

    pd.DataFrame(test_frames_with_labels_new).to_csv('created_tables/seq_to_seq_test_withlabels.csv', sep='\t')

def seq_to_seq4(seq_id_map, train_frames, test_frames):

    train_frames_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2'])
    for table_index, row in train_frames.iterrows():
        human = row['protein_sequence1']
        virus = row['protein_sequence2']
        interaction = row['interaction']
        # Change sequence with sequence id
        for row_index, row in seq_id_map.iterrows():
            if row['sequence'] == human:
                if row_index == 0:
                    print('is zero')
                else:
                    hum_id = row_index
                    train_frames.at[table_index, 'protein_sequence1'] = hum_id
            if row['sequence'] == virus:
                if row_index == 0:
                    print('is zero')
                else:
                    vir_id = row_index
                    train_frames.at[table_index, 'protein_sequence2'] = vir_id


        train_frames_new = train_frames_new.append({'protein_sequence1': hum_id, 'protein_sequence2': vir_id, },
                                                       ignore_index=True)

    print(train_frames)
    pd.DataFrame(train_frames).to_csv('created_tables/seq_to_seq_train_withlabels.csv', sep='\t')

    test_frames_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2'])
    for table_index, row in test_frames.iterrows():
        human = row['protein_sequence1']
        virus = row['protein_sequence2']
        interaction = row['interaction']
        # Change sequence with sequence id
        for row_index, row in seq_id_map.iterrows():
            if row['sequence'] == human:
                if row_index == 0:
                    print('is zero')
                else:
                    hum_id = row_index
                    test_frames.at[table_index, 'protein_sequence1'] = hum_id
            if row['sequence'] == virus:
                if row_index == 0:
                    print('is zero')
                else:
                    vir_id = row_index
                    test_frames.at[table_index, 'protein_sequence2'] = vir_id


        test_frames_new = test_frames_new.append({'protein_sequence1': hum_id, 'protein_sequence2': vir_id},
                                                     ignore_index=True)

    print(test_frames)
    pd.DataFrame(test_frames).to_csv('created_tables/seq_to_seq_test_withlabels.csv', sep='\t')

def train_test_to_csv(train_frames, test_frames):

    train_frames.to_csv('created_tables/train_frames_original.csv', sep='\t')
    test_frames.to_csv('created_tables/test_frames_original.csv', sep='\t')

def get_seqvec_of_sequences():

    # Get seq_id_map
    seq_id_map = pd.read_csv('created_tables/setup3_seqvec/seq_id_map.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id", "sequence"]  # set our own names for the columns
        )
    seq_id_map = seq_id_map.iloc[1:, :]

    # Create sequence to feature (embedding) map
    sequence_features = pd.DataFrame(columns=['id', 'embedding'])

    print(seq_id_map)

    # 1. SeqVec
    # Load pre-trained model:
    model_dir = Path('seqvecmodel')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    seqvec = ElmoEmbedder(options, weights, cuda_device=0)  # cuda_device=-1 for CPU
    print('seqvec model loaded.')
    print('Protein:', seq_id_map.iloc[4]['sequence'])
    seq = seq_id_map.iloc[4]['sequence']  # your amino acid sequence
    embedding = seqvec.embed_sentence(list(seq))  # List-of-Lists with shape [3,L,1024]
    protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)

    print('Protein embedding:', protein_embd)

    count = 0

    #seq_id_map = seq_id_map[2000:2740]

    for table_index, row in seq_id_map.iterrows():

        print(count)

        sequence = row['sequence']
        id = row['id']

        embedding = seqvec.embed_sentence(list(sequence))  # List-of-Lists with shape [3,L,1024]
        protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)
        protein_embd = protein_embd.cpu().detach().numpy()

        row_to_add = {'id': id, 'embedding': protein_embd}
        sequence_features = sequence_features.append(row_to_add, ignore_index=True)

        count+=1

    print(sequence_features)
    sequence_features.to_pickle('created_tables/setup3_seqvec/seqid_to_embedding_all.pkl')

    #sequence_features.to_csv('created_tables/setup3_seqvec/seqid_to_embedding_0_500.csv', sep='\t')
    return sequence_features

def arrange_seqvec_feature_files():

    all = pd.read_pickle('created_tables/setup3_seqvec/seqid_to_embedding_all.pkl')
    print(all)

    # 2740 x 1024
    seq_to_emb = np.zeros((2741, 1024))

    for row_index, row in all.iterrows():
        embedding = row['embedding']
        for col_index in range(1024):
            seq_to_emb[row_index][col_index] = embedding[col_index]

    print(seq_to_emb)
    pd.DataFrame(seq_to_emb).to_csv('created_tables/setup3_seqvec/seq_to_embedding_features.csv', sep='\t')

def arrange_seqvec_feature_files2():

    all = pd.read_pickle('created_tables/setup3_seqvec/seqid_to_embedding_all.pkl')
    print(all)

    # 2740 x 1024
    seq_to_emb = np.zeros((2741, 1))

    for row_index, row in all.iterrows():
        embedding = row['embedding']
        embedding_sum = 0
        for col_index in range(1024):
            embedding_sum += embedding[col_index]

        seq_to_emb[row_index][0] = embedding_sum/1024

    print(seq_to_emb)
    pd.DataFrame(seq_to_emb).to_csv('created_tables/setup3_seqvec/seq_to_embedding_features_mean.csv', sep='\t')

def arrange_tfidf_files():
    word_tfidf = pd.read_csv('created_tables/tfidf/word_tfidf.csv',
                             sep="\t",  # tab-separated
                             header=None,  # no heading row
                             names=["id", "tfidf"]  # set our own names for the columns
                             )

    word_tfidf = word_tfidf.iloc[1:, :]
    print(word_tfidf)

    word_tfidf = word_tfidf.sort_values(by=["tfidf"], ascending=False)
    print(word_tfidf)

    protein_word_tfidf = pd.read_csv(
        'created_tables/tfidf/seq_to_word_tfidfvalues.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    protein_word_tfidf = protein_word_tfidf.iloc[1:, :]
    protein_word_tfidf = protein_word_tfidf.iloc[:, 1:]
    print(protein_word_tfidf)

    # Get all 500 id in word_tfidf:
    biggest = word_tfidf[0:500]
    ids = biggest['id']
    ids += 1
    ids = ids.values.tolist()

    remove = word_tfidf[500:4761]
    remove = remove['id']
    remove += 1
    remove = remove.values.tolist()

    print(ids)
    print(remove)

    for id in remove:
        protein_word_tfidf.pop(id)

    protein_word_tfidf.to_csv('created_tables/tfidf/seq_to_embedding_filtered.csv', sep='\t')
    print(protein_word_tfidf)

def main():

    train_frames, test_frames = read_datasets()
    tokenizer = train_tokenizer()

    print(train_frames)
    print(test_frames)

    # Unique sequences to id mapping
    seq_id_map = seq_to_id_with_types(train_frames, test_frames)

    # Unique words to id mapping
    word_id_map = word_to_id(seq_id_map, tokenizer, train_frames, test_frames)

    # Prot sequence id to word
    seq_to_word_map = seq_to_word2(seq_id_map, word_id_map, tokenizer)

    # Prot sequence id to sequence id
    seq_to_seq_map = seq_to_seq3(seq_id_map, train_frames, test_frames)

    #sequence_features = get_seqvec_of_sequences()
    #arrange_seqvec_feature_files2()

    #arrange_tfidf_files()



if __name__ == '__main__':
    main()