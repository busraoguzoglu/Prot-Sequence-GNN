from stellargraph import StellarGraph
import pandas as pd
import numpy as np
from collections import OrderedDict
from tokenizers import pre_tokenizers
from tokenizers import ByteLevelBPETokenizer

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

def word_to_id(seq_id_map, tokenizer):

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
    return df

def seq_to_word(seq_id_map, word_id_map, tokenizer):

    seq_to_word = np.empty((len(seq_id_map), len(word_id_map)))
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
                seq_to_word_frame = seq_to_word_frame.append({'source': row_index, 'target': col_index},
                                                           ignore_index=True)

        count+=1



    # Write to file
    seq_id_map.to_csv('created_tables/seq_id_map.csv', sep='\t')
    word_id_map.to_csv('created_tables/word_id_map.csv', sep='\t')

    seq_to_word_frame.to_csv('created_tables/seq_to_word.csv', sep='\t')

def seq_to_seq(seq_id_map, train_frames, test_frames):

    seq_to_seq = np.empty((len(seq_id_map), len(seq_id_map)))

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
    seq_to_seq = np.empty((len(seq_id_map), len(seq_id_map)))

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

    train_frames_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2'])
    train_frames_with_labels_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2', 'interaction'])

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

        if int(interaction) == 1:
            train_frames_new = train_frames_new.append({'protein_sequence1': hum_id, 'protein_sequence2': vir_id}, ignore_index=True)

        train_frames_with_labels_new = train_frames_with_labels_new.append({'protein_sequence1': hum_id, 'protein_sequence2': vir_id, 'interaction': interaction},
                                                   ignore_index=True)

    print(train_frames_new)
    pd.DataFrame(train_frames_new).to_csv('created_tables/seq_to_seq_train.csv', sep='\t')
    pd.DataFrame(train_frames_with_labels_new).to_csv('created_tables/seq_to_seq_train_withlabels.csv', sep='\t')


    test_frames_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2'])
    test_frames_with_labels_new = pd.DataFrame(columns=['protein_sequence1', 'protein_sequence2', 'interaction'])

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
                    train_frames.at[table_index, 'protein_sequence1'] = hum_id
            if row['sequence'] == virus:
                if row_index == 0:
                    print('is zero')
                else:
                    vir_id = row_index
                    train_frames.at[table_index, 'protein_sequence2'] = vir_id

        if int(interaction) == 1:
            test_frames_new = test_frames_new.append({'protein_sequence1': hum_id, 'protein_sequence2': vir_id},
                                                       ignore_index=True)

        test_frames_with_labels_new = test_frames_with_labels_new.append(
            {'protein_sequence1': hum_id, 'protein_sequence2': vir_id, 'interaction': interaction},
            ignore_index=True)


    print(test_frames_new)
    pd.DataFrame(test_frames_new).to_csv('created_tables/seq_to_seq_test.csv', sep='\t')
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

def main():

    train_frames, test_frames = read_datasets()
    tokenizer = train_tokenizer()

    # Unique sequences to id mapping
    seq_id_map = seq_to_id(train_frames, test_frames)

    # Unique words to id mapping
    word_id_map = word_to_id(seq_id_map, tokenizer)

    # Prot sequence id to word
    seq_to_word_map = seq_to_word2(seq_id_map, word_id_map, tokenizer)

    # Prot sequence id to sequence id
    seq_to_seq_map = seq_to_seq3(seq_id_map, train_frames, test_frames)

if __name__ == '__main__':
    main()