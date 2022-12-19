import numpy
import pandas as pd
from stellargraph import datasets
from IPython.display import display, HTML
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
from Levenshtein import distance

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score

from stellargraph import StellarGraph
from stellargraph import IndexedArray
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Dense, Activation, Dropout

def get_predictions(y_pred_value):

    predictions = []

    print('preds:', y_pred_value)

    for value in y_pred_value:
        if value >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

def create_graph():

    # Graph will only train nodes will be created.

    # Prot-word edges:
    protein_word_content = pd.read_csv(
        'created_tables/clean_metapath/seq_to_word_train_tfidf_filtered_novalid2.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    protein_word_content.drop(columns=protein_word_content.columns[0], axis=1, inplace=True)
    protein_word_content = protein_word_content.iloc[1:, :]
    protein_word_content.columns = ['source', 'target']
    protein_word_content = protein_word_content.astype(float)
    protein_word_content = protein_word_content.astype(int)
    print('Protein word content: (edges)', protein_word_content)

    # Prot-prot interaction edges:
    protein_interactions_train_labels = pd.read_csv(
        'created_tables/clean_metapath/seq_to_seq_train_withlabels_novalid2.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target", "interaction_type"]  # set our own names for the columns
    )

    # Need to make all edges have unique id
    train_labels_index = []
    #9753
    #8778
    for i in range(9519):
        train_labels_index.append(i + len(protein_word_content) + 1)

    protein_interactions_train_labels = protein_interactions_train_labels.iloc[1:, :]
    protein_interactions_train_labels['index'] = train_labels_index
    protein_interactions_train_labels.set_index(['index'], inplace=True)
    protein_interactions_train_labels = protein_interactions_train_labels.astype(int)

    # Word-word edges:
    # First edge id should be 160517 + 1

    word_word_edges = pd.read_csv(
        'created_tables/clean_metapath/word_to_word_edges.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )

    word_word_edges = word_word_edges.iloc[1:, :]

    # Need to make all edges have unique id
    word_word_edges_index = []
    for i in range(len(word_word_edges)):
        word_word_edges_index.append(i + 160517 + 1)

    word_word_edges['index'] = word_word_edges_index
    word_word_edges.set_index(['index'], inplace=True)
    word_word_edges = word_word_edges.astype(int)
    print('word_word_edges:', word_word_edges)


    #################################################################################
    # Needleman Wunsch #
    #################################################################################

    # Human
    hum_similarity = pd.read_csv(
        "created_tables/similarity_data/protein_similarities_human.csv",
        header=None,  # no heading row
    )

    # Make id to index map for this:
    id_index_map_hum = hum_similarity.iloc[:, :1]
    id_index_map_hum.drop(index=0, inplace=True)
    id_index_map_hum.index -= 1
    id_index_map_hum.rename(columns={id_index_map_hum.columns[0]: "id"}, inplace=True)
    id_index_map_hum['id'] = id_index_map_hum['id'].astype('int')

    # Make dataframe into a matrix:
    hum_similarity.drop(index=0, inplace=True)
    hum_similarity.drop(columns=hum_similarity.columns[0], axis=1, inplace=True)
    arr_hum = hum_similarity.to_numpy()

    # Virus
    vir_similarity = pd.read_csv(
        "created_tables/similarity_data/protein_similarities_vir.csv",
        header=None,  # no heading row
    )

    # Make id to index map for this:
    id_index_map_vir = vir_similarity.iloc[:, :1]
    id_index_map_vir.drop(index=0, inplace=True)
    id_index_map_vir.index -= 1
    id_index_map_vir.rename(columns={id_index_map_vir.columns[0]: "id"}, inplace=True)
    id_index_map_vir['id'] = id_index_map_vir['id'].astype('int')

    # Make dataframe into a matrix:
    vir_similarity.drop(index=0, inplace=True)
    vir_similarity.drop(columns=vir_similarity.columns[0], axis=1, inplace=True)
    arr_vir = vir_similarity.to_numpy()

    # Get train sequences
    hum_ids = pd.read_csv(
        'created_tables/clean_metapath/all_seq_id_human_train.csv',
        sep="\t",  # tab-separated
    )

    vir_ids = pd.read_csv(
        'created_tables/clean_metapath/all_seq_id_virus_train.csv',
        sep="\t",  # tab-separated
    )

    hum = hum_ids['id'].tolist()
    vir = vir_ids['id'].tolist()
    similarity_cutoff = 50
    source_hum = []
    target_hum = []
    source_vir = []
    target_vir = []

    for i in range(len(hum)):
        id1 = hum[i]
        ind1 = id_index_map_hum.index[id_index_map_hum['id'] == id1].tolist()[0]
        for y in range(i+1, len(hum)):
            id2 = hum[y]
            ind2 = id_index_map_hum.index[id_index_map_hum['id'] == id2].tolist()[0]
            sim = arr_hum[ind2][ind1]
            if sim > similarity_cutoff:
                source_hum.append(id1)
                target_hum.append(id2)

    for i in range(len(vir)):
        id1 = vir[i]
        ind1 = id_index_map_vir.index[id_index_map_vir['id'] == id1].tolist()[0]
        for y in range(i+1, len(vir)):
            id2 = vir[y]
            ind2 = id_index_map_vir.index[id_index_map_vir['id'] == id2].tolist()[0]
            if arr_vir[ind2][ind1] > similarity_cutoff:
                source_vir.append(id1)
                target_vir.append(id2)

    columns = ['source', 'target']
    df_hum = pd.DataFrame(list(zip(source_hum, target_hum)),
                      columns=columns)
    df_vir = pd.DataFrame(list(zip(source_vir, target_vir)),
                          columns=columns)

    # Need to make all edges have unique id
    hum_similarity_edges_index = []
    edge_start = word_word_edges_index[len(word_word_edges_index) - 1]

    for i in range(len(df_hum)):
        hum_similarity_edges_index.append(i + edge_start + 1)

    df_hum['index'] = hum_similarity_edges_index
    df_hum.set_index(['index'], inplace=True)
    df_hum = df_hum.astype(int)
    print('hum_similarity_edges:', df_hum)

    vir_similarity_edges_index = []
    edge_start = hum_similarity_edges_index[len(hum_similarity_edges_index) - 1]

    for i in range(len(df_vir)):
        vir_similarity_edges_index.append(i + edge_start + 1)

    df_vir['index'] = vir_similarity_edges_index
    df_vir.set_index(['index'], inplace=True)
    df_vir = df_vir.astype(int)
    print('vir_similarity_edges:', df_vir)

    #########################################################################################
    ### NODES ####
    #########################################################################################

    # Get human, virus and word indexes:

    word_ids = pd.read_csv(
        'created_tables/clean_metapath/all_word_id_train.csv',
        sep="\t",  # tab-separated
    )
    word = word_ids['id'].tolist()

    # Create nodes (hum-vir-word)
    hum_nodes = IndexedArray(index=hum)
    vir_nodes = IndexedArray(index=vir)
    word_nodes = IndexedArray(index=word)

    print('protein_interactions_train_labels (interaction edges):', protein_interactions_train_labels)

    # Get %10 of this as validation set.
    # For all sequences in this, remove corresponding nodes&edges

    interacts = protein_interactions_train_labels[protein_interactions_train_labels['interaction_type'] == 1]
    not_interacts = protein_interactions_train_labels[protein_interactions_train_labels['interaction_type'] == 0]

    protein_graph_train = StellarGraph({"human": hum_nodes, "virus": vir_nodes, "word": word_nodes},
                                       edges={'contains': protein_word_content, 'interacts': interacts,
                                              'not_interacts': not_interacts, 'pmi': word_word_edges,
                                              'hum_similarity': df_hum,
                                              'vir_similarity': df_vir})

    print(protein_graph_train.info())
    return protein_graph_train

def get_train_node_embeddings(g):

    walk_length = 250  # maximum length of a random walk to use throughout this notebook

    # specify the metapath schemas as a list of lists of node types.

    # Metapath for whole graph
    metapaths = [
        ["human", "human"],
        ["virus", "virus"],
        ["human", "word", "human"],
        ["virus", "word", "virus"],
        ["human", "virus", "human"],
        ["virus", "human", "virus"],
        ["human", "virus", "human", "word", "human"],
        ["virus", "human", "virus", "word", "virus"],
        ["human", "word", "human", "virus", "human"],
        ["virus", "word", "virus", "human", "virus"],
        ["human", "word", "word", "human"],
        ["virus", "word", "word", "virus"]
    ]

    # No word nodes
    """
    metapaths = [
        ["human", "human"],
        ["virus", "virus"],
        ["human", "virus", "human"],
        ["virus", "human", "virus"],
        ["human", "virus", "virus", "human"],
        ["virus", "human", "human", "virus"],
    ]
    """

    # No sequence similarity

    """
    metapaths = [
        ["human", "word", "human"],
        ["virus", "word", "virus"],
        ["human", "virus", "human"],
        ["virus", "human", "virus"],
        ["human", "virus", "human", "word", "human"],
        ["virus", "human", "virus", "word", "virus"],
        ["human", "word", "human", "virus", "human"],
        ["virus", "word", "virus", "human", "virus"],
        ["human", "word", "word", "human"],
        ["virus", "word", "word", "virus"]
    ]
    """


    # Create the random walker
    rw = UniformRandomMetaPathWalk(g)

    walks = rw.run(
        nodes=list(g.nodes()),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=3,  # number of random walks per root node
        metapaths=metapaths,  # the metapaths
    )

    print("Number of random walks: {}".format(len(walks)))
    model = Word2Vec(walks, vector_size=256, window=10, min_count=0, sg=1, workers=2, epochs=1)

    # Retrieve node embeddings and corresponding subjects
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [g.node_type(node_id) for node_id in node_ids]

    # Sorting does not seem to be working...
    node_ids, node_embeddings = zip(*sorted(zip(node_ids, node_embeddings)))

    return node_embeddings[0:2648]

def get_train_embeddings(node_embeddings_train):

    # Get train set:
    protein_interactions_train = pd.read_csv(
        'created_tables/clean_metapath/seq_to_seq_train_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["protein_sequence1", "protein_sequence2", "label"]  # set our own names for the columns
    )
    protein_interactions_train = protein_interactions_train.iloc[1:, :]
    protein_interactions_train = protein_interactions_train.astype(int)

    # Change ids with embeddings:
    train_data = pd.DataFrame(columns=['embedding', 'label'])

    for row_index, row in protein_interactions_train.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        label = row['label']
        sequence1emb = node_embeddings_train[sequence1]
        sequence2emb = node_embeddings_train[sequence2]
        sequences = numpy.concatenate((sequence1emb, sequence2emb), axis=0)
        new_row = {'embedding': sequences, 'label': label}
        train_data = train_data.append(new_row, ignore_index=True)

    print('train data:', train_data)
    return train_data

def get_valid_embeddings(node_embeddings_train):

    # Get valid set and create test embeddings
    # Test embeddings will be calculated using the similarity of
    # sequences between train and test sets
    protein_interactions_test = pd.read_csv(
        'created_tables/clean_metapath/seq_to_seq_valid_withlabels2.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["protein_sequence1", "protein_sequence2", "label"]  # set our own names for the columns
    )
    protein_interactions_test = protein_interactions_test.iloc[1:, :]
    protein_interactions_test = protein_interactions_test.astype(int)
    print(protein_interactions_test)

    # Get seq-id-map train and seq-id-map test:
    seq_to_id_train = pd.read_csv(
        'created_tables/clean_metapath/seq_id_map_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id", "sequence", "type"]  # set our own names for the columns
    )
    seq_to_id_train = seq_to_id_train.iloc[1:, :]
    seq_to_id_train['id'] = seq_to_id_train['id'].astype('int')
    seq_to_id_train.set_index(['id'], inplace=True)
    print(seq_to_id_train)

    seq_to_id_test = pd.read_csv(
        'created_tables/clean_metapath/seq_id_map_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id", "sequence", "type"]  # set our own names for the columns
    )
    seq_to_id_test = seq_to_id_test.iloc[1:, :]
    seq_to_id_test['id'] = seq_to_id_test['id'].astype('int')
    seq_to_id_test.set_index(['id'], inplace=True)
    print('seq to id test:', seq_to_id_test)

    ###############################################################################################
    ############# Read similarity data, create matrices and id maps ###############################
    ###############################################################################################

    # Human
    hum_similarity = pd.read_csv(
        "created_tables/similarity_data/protein_similarities_human.csv",
        header=None,  # no heading row
    )

    # Make id to index map for this:
    id_index_map_hum = hum_similarity.iloc[:, :1]
    id_index_map_hum.drop(index=0, inplace=True)
    id_index_map_hum.index -= 1
    id_index_map_hum.rename(columns={id_index_map_hum.columns[0]: "id"}, inplace=True)
    id_index_map_hum['id'] = id_index_map_hum['id'].astype('int')

    # Make dataframe into a matrix:
    hum_similarity.drop(index=0, inplace=True)
    hum_similarity.drop(columns=hum_similarity.columns[0], axis=1, inplace=True)
    arr_hum = hum_similarity.to_numpy()

    # Virus
    vir_similarity = pd.read_csv(
        "created_tables/similarity_data/protein_similarities_vir.csv",
        header=None,  # no heading row
    )

    # Make id to index map for this:
    id_index_map_vir = vir_similarity.iloc[:, :1]
    id_index_map_vir.drop(index=0, inplace=True)
    id_index_map_vir.index -= 1
    id_index_map_vir.rename(columns={id_index_map_vir.columns[0]: "id"}, inplace=True)
    id_index_map_vir['id'] = id_index_map_vir['id'].astype('int')

    # Make dataframe into a matrix:
    vir_similarity.drop(index=0, inplace=True)
    vir_similarity.drop(columns=vir_similarity.columns[0], axis=1, inplace=True)
    arr_vir = vir_similarity.to_numpy()

    # Change ids with embeddings:
    # Go over test data, calculate similarities, and calculate embeddings:
    test_data = pd.DataFrame(columns=['embedding', 'label'])

    for row_index, row in protein_interactions_test.iterrows():

        # +10000 for test id
        sequence1id = row['protein_sequence1']
        sequence2id = row['protein_sequence2']

        index_seq1 = id_index_map_hum.index[id_index_map_hum['id'] == sequence1id].tolist()[0]
        index_seq2 = id_index_map_vir.index[id_index_map_vir['id'] == sequence2id].tolist()[0]

        label = row['label']

        sequence1_similarity_values = []
        sequence2_similarity_values = []
        sequence1_similarity_trainseqid = []
        sequence2_similarity_trainseqid = []

        #########################################################################

        # For each sequence, find the most similar 5 train sequences:

        # ids go 0 to 2647

        for id in range(0, 2648):

            hum = 0
            # Need to find if this id is human or virus:
            index_id_list = id_index_map_hum.index[id_index_map_hum['id'] == id].tolist()
            if index_id_list:
                index_id = index_id_list[0]
                hum = 1
            else:
                index_id = id_index_map_vir.index[id_index_map_vir['id'] == id].tolist()[0]

            # get the similarity value for given ids:
            if hum:
                seq1sim = arr_hum[index_seq1][index_id]
                sequence1_similarity_values.append(seq1sim)
                sequence1_similarity_trainseqid.append(id)
            else:
                seq2sim = arr_vir[index_seq2][index_id]
                sequence2_similarity_values.append(seq2sim)
                sequence2_similarity_trainseqid.append(id)

        # Sort similarity values, get most similar 5 ids for seq1 and seq2
        sequence1_similarity_values, sequence1_similarity_trainseqid = zip(
            *sorted(zip(sequence1_similarity_values, sequence1_similarity_trainseqid), reverse=True))
        sequence2_similarity_values, sequence2_similarity_trainseqid = zip(
            *sorted(zip(sequence2_similarity_values, sequence2_similarity_trainseqid), reverse=True))

        # Now get the embedding of these sequences for each,
        # and get the average. This will give embedding of the test sequence:

        # For sequence1:
        # ar1 = node_embeddings_train[sequence1_similarity_trainseqid[0]]
        # ar2 = node_embeddings_train[sequence1_similarity_trainseqid[1]]
        # seq1emb = ((2*ar1) + ar2) / 3
        # seq1emb = ar1

        # Get more than %90 similar embeddings and take average of them:
        # At most 10 is enough
        avg_list = []
        sum = node_embeddings_train[sequence1_similarity_trainseqid[0]]
        for i in range(1, 10):
            if sequence1_similarity_values[i] > 99:
                avg_list.append(node_embeddings_train[sequence1_similarity_trainseqid[i]])
                sum += node_embeddings_train[sequence1_similarity_trainseqid[i]]
            else:
                continue

        diff_size = 0
        if len(avg_list) > 1:
            seq1emb = (sum - 100) / (len(avg_list))
            diff_size +=1

        else:
            seq1emb = sum / (len(avg_list) + 1)

        # For sequence2:
        # ar1 = node_embeddings_train[sequence2_similarity_trainseqid[0]]
        # ar2 = node_embeddings_train[sequence2_similarity_trainseqid[1]]
        # seq2emb = ((2*ar1) + ar2) / 3
        # seq2emb = ar1

        # Get more than %90 similar embeddings and take average of them:
        # At most 10 is enough
        avg_list = []
        sum = node_embeddings_train[sequence2_similarity_trainseqid[0]]
        for i in range(1, 10):
            if sequence2_similarity_values[i] > 99:
                avg_list.append(node_embeddings_train[sequence2_similarity_trainseqid[i]])
                sum += node_embeddings_train[sequence2_similarity_trainseqid[i]]
            else:
                continue

        diff_size = 0
        if len(avg_list) > 1:
            seq2emb = (sum-100) / (len(avg_list))
            diff_size += 1

        else:
            seq2emb = sum / (len(avg_list) + 1)

        sequences_appended = numpy.concatenate((seq1emb, seq2emb), axis=0)
        new_row = {'embedding': sequences_appended, 'label': label}
        test_data = test_data.append(new_row, ignore_index=True)

    print('valid data:', test_data)
    return test_data


def get_test_embeddings(node_embeddings_train):

    # Get test set and create test embeddings
    # Test embeddings will be calculated using the similarity of
    # sequences between train and test sets
    protein_interactions_test = pd.read_csv(
        'created_tables/clean_metapath/seq_to_seq_test_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["protein_sequence1", "protein_sequence2", "label"]  # set our own names for the columns
    )
    protein_interactions_test = protein_interactions_test.iloc[1:, :]
    protein_interactions_test = protein_interactions_test.astype(int)
    print(protein_interactions_test)

    # Get seq-id-map train and seq-id-map test:
    seq_to_id_train = pd.read_csv(
        'created_tables/clean_metapath/seq_id_map_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id", "sequence", "type"]  # set our own names for the columns
    )
    seq_to_id_train = seq_to_id_train.iloc[1:, :]
    seq_to_id_train['id'] = seq_to_id_train['id'].astype('int')
    seq_to_id_train.set_index(['id'], inplace=True)
    print(seq_to_id_train)

    seq_to_id_test = pd.read_csv(
        'created_tables/clean_metapath/seq_id_map_test.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["id", "sequence", "type"]  # set our own names for the columns
    )
    seq_to_id_test = seq_to_id_test.iloc[1:, :]
    seq_to_id_test['id'] = seq_to_id_test['id'].astype('int')
    seq_to_id_test.set_index(['id'], inplace=True)
    print('seq to id test:', seq_to_id_test)

    ###############################################################################################
    ############# Read similarity data, create matrices and id maps ###############################
    ###############################################################################################

    # Human
    hum_similarity = pd.read_csv(
        "created_tables/similarity_data/protein_similarities_human.csv",
        header=None,  # no heading row
    )

    # Make id to index map for this:
    id_index_map_hum = hum_similarity.iloc[:, :1]
    id_index_map_hum.drop(index=0, inplace=True)
    id_index_map_hum.index -= 1
    id_index_map_hum.rename(columns={id_index_map_hum.columns[0]: "id"}, inplace=True)
    id_index_map_hum['id'] = id_index_map_hum['id'].astype('int')

    # Make dataframe into a matrix:
    hum_similarity.drop(index=0, inplace=True)
    hum_similarity.drop(columns=hum_similarity.columns[0], axis=1, inplace=True)
    arr_hum = hum_similarity.to_numpy()

    # Virus
    vir_similarity = pd.read_csv(
        "created_tables/similarity_data/protein_similarities_vir.csv",
        header=None,  # no heading row
    )

    # Make id to index map for this:
    id_index_map_vir = vir_similarity.iloc[:, :1]
    id_index_map_vir.drop(index=0, inplace=True)
    id_index_map_vir.index -= 1
    id_index_map_vir.rename(columns={id_index_map_vir.columns[0]: "id"}, inplace=True)
    id_index_map_vir['id'] = id_index_map_vir['id'].astype('int')

    # Make dataframe into a matrix:
    vir_similarity.drop(index=0, inplace=True)
    vir_similarity.drop(columns=vir_similarity.columns[0], axis=1, inplace=True)
    arr_vir = vir_similarity.to_numpy()

    # Change ids with embeddings:
    # Go over test data, calculate similarities, and calculate embeddings:
    test_data = pd.DataFrame(columns=['embedding', 'label'])

    for row_index, row in protein_interactions_test.iterrows():

        # +10000 for test id
        sequence1id = row['protein_sequence1'] + 10000
        sequence2id = row['protein_sequence2'] + 10000

        index_seq1 = id_index_map_hum.index[id_index_map_hum['id'] == sequence1id].tolist()[0]
        index_seq2 = id_index_map_vir.index[id_index_map_vir['id'] == sequence2id].tolist()[0]

        label = row['label']

        sequence1_similarity_values = []
        sequence2_similarity_values = []
        sequence1_similarity_trainseqid = []
        sequence2_similarity_trainseqid = []

        #########################################################################

        # For each sequence, find the most similar 5 train sequences:

        # ids go 0 to 2647

        for id in range (0, 2648):

            hum = 0
            # Need to find if this id is human or virus:
            index_id_list = id_index_map_hum.index[id_index_map_hum['id'] == id].tolist()
            if index_id_list:
                index_id = index_id_list[0]
                hum = 1
            else:
                index_id = id_index_map_vir.index[id_index_map_vir['id'] == id].tolist()[0]

            # get the similarity value for given ids:
            if hum:
                seq1sim = arr_hum[index_seq1][index_id]
                sequence1_similarity_values.append(seq1sim)
                sequence1_similarity_trainseqid.append(id)
            else:
                seq2sim = arr_vir[index_seq2][index_id]
                sequence2_similarity_values.append(seq2sim)
                sequence2_similarity_trainseqid.append(id)

        # Sort similarity values, get most similar 5 ids for seq1 and seq2
        sequence1_similarity_values, sequence1_similarity_trainseqid = zip(
            *sorted(zip(sequence1_similarity_values, sequence1_similarity_trainseqid), reverse=True))
        sequence2_similarity_values, sequence2_similarity_trainseqid = zip(
            *sorted(zip(sequence2_similarity_values, sequence2_similarity_trainseqid), reverse= True))


        # Now get the embedding of these sequences for each,
        # and get the average. This will give embedding of the test sequence:

        # For sequence1:
        #ar1 = node_embeddings_train[sequence1_similarity_trainseqid[0]]
        #ar2 = node_embeddings_train[sequence1_similarity_trainseqid[1]]
        #seq1emb = ((2*ar1) + ar2) / 3
        #seq1emb = ar1

        # Get more than %90 similar embeddings and take average of them:
        # At most 10 is enough
        avg_list = []
        sum = node_embeddings_train[sequence1_similarity_trainseqid[0]]
        for i in range(1, 10):
            if sequence1_similarity_values[i] > 80:
                avg_list.append(node_embeddings_train[sequence1_similarity_trainseqid[i]])
                sum += node_embeddings_train[sequence1_similarity_trainseqid[i]]
            else:
                continue

        seq1emb = sum/(len(avg_list)+1)


        # For sequence2:
        #ar1 = node_embeddings_train[sequence2_similarity_trainseqid[0]]
        #ar2 = node_embeddings_train[sequence2_similarity_trainseqid[1]]
        #seq2emb = ((2*ar1) + ar2) / 3
        #seq2emb = ar1

        # Get more than %90 similar embeddings and take average of them:
        # At most 10 is enough
        avg_list = []
        sum = node_embeddings_train[sequence2_similarity_trainseqid[0]]
        for i in range(1, 10):
            if sequence2_similarity_values[i] > 99:
                avg_list.append(node_embeddings_train[sequence2_similarity_trainseqid[i]])
                sum += node_embeddings_train[sequence2_similarity_trainseqid[i]]
            else:
                continue

        seq2emb = sum / (len(avg_list) + 1)

        sequences_appended = numpy.concatenate((seq1emb, seq2emb), axis=0)
        new_row = {'embedding': sequences_appended, 'label': label}
        test_data = test_data.append(new_row, ignore_index=True)

    print('test data:', test_data)
    return test_data

def main():

    # Create graph (protein sequence-word graph)
    g = create_graph()

    # Use metapath2vec to get node embeddings
    # This contain train sequence embeddings sorted by id
    node_embeddings_train = get_train_node_embeddings(g)

    # Get train/test datasets for classifier
    train_data = get_train_embeddings(node_embeddings_train)
    test_data = get_test_embeddings(node_embeddings_train)
    valid_data = get_valid_embeddings(node_embeddings_train)

    #######################################################
    #### Training ####
    #######################################################

    # Training the classifier #
    train_X = train_data[['embedding']]
    test_X = test_data[['embedding']]
    valid_X = valid_data[['embedding']]
    train_labels = train_data[['label']]
    test_labels = test_data[['label']]
    valid_labels = valid_data[['label']]

    train_X, train_Y = shuffle(train_X, train_labels, random_state=0)
    test_X, test_Y = shuffle(test_X, test_labels, random_state=0)
    valid_X, valid_Y = shuffle(valid_X, valid_labels, random_state=0)

    train_X_list = []
    for row_index, row in train_X.iterrows():
        emb = row['embedding']
        train_X_list.append(emb)

    test_X_list = []
    for row_index, row in test_X.iterrows():
        emb = row['embedding']
        test_X_list.append(emb)

    valid_X_list = []
    for row_index, row in valid_X.iterrows():
        emb = row['embedding']
        valid_X_list.append(emb)

    train_Y_list = []
    for row_index, row in train_Y.iterrows():
        lab = row['label']
        train_Y_list.append(lab)

    test_Y_list = []
    for row_index, row in test_Y.iterrows():
        lab = row['label']
        test_Y_list.append(lab)

    valid_Y_list = []
    for row_index, row in valid_Y.iterrows():
        lab = row['label']
        valid_Y_list.append(lab)

    train_X = train_X_list
    test_X = test_X_list
    valid_X = valid_X_list
    train_Y = train_Y_list
    test_Y = test_Y_list
    valid_Y = valid_Y_list


    # Classification:

    # 2- Tensorflow:
    test_X = numpy.array(test_X)
    test_Y = numpy.array(test_Y)

    valid_X = numpy.array(valid_X)
    valid_Y = numpy.array(valid_Y)

    train_X = numpy.asarray(train_X).astype(numpy.float32)
    train_Y = numpy.asarray(train_Y).astype(numpy.float32)

    epochs = 300
    batch_size = 128

    recalls, specs, npvs, accs, precs, mccs, aucs, f1s = [], [], [], [], [], [], [], []

    for i in range(10):
        # Model:
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(Input(shape=(512,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        # model.fit(train_X, train_Y, validation_data=(val_X, val_Y), batch_size=batch_size, epochs=epochs, verbose=0)
        model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=0)

        # Valid result
        y_true = valid_Y
        y_pred_label = model.predict(valid_X)

        # Get test result later
        #y_true = test_Y
        #y_pred_label = model.predict(test_X)
        y_pred_label = get_predictions(y_pred_label)

        y_pred_label = list(y_pred_label)
        y_true = list(y_true)

        print('y_pred_label', y_pred_label)
        print('y_true', y_true)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
        recall = recall_score(y_true, y_pred_label)
        spec = tn / (tn + fp)
        npv = tn / (tn + fn)
        acc = accuracy_score(y_true, y_pred_label)
        prec = precision_score(y_true, y_pred_label)
        mcc = matthews_corrcoef(y_true, y_pred_label)
        auc = roc_auc_score(y_true, y_pred_label)
        f1 = 2 * prec * recall / (prec + recall)
        print(
            "Sensitivity: %.4f, Specificity: %.4f, Accuracy: %.4f, PPV: %.4f, NPV: %.4f, MCC: %.4f, AUC: %.4f, F1: ROCAUC: %.4f" \
            % (recall * 100, spec * 100, acc * 100, prec * 100, npv * 100, mcc, auc, f1 * 100))
        recalls.append(recall)
        specs.append(spec)
        npvs.append(npv)
        accs.append(acc)
        precs.append(prec)
        mccs.append(mcc)
        aucs.append(auc)
        f1s.append(f1)

    count = 10
    recall = sum(recalls) / count
    spec = sum(specs) / count
    npv = sum(npvs) / count
    acc = sum(accs) / count
    prec = sum(precs) / count
    mcc = sum(mccs) / count
    auc = sum(aucs) / count
    f1 = sum(f1s) / count

    print("Sensitivity: %.4f, Specificity: %.4f, Accuracy: %.4f, PPV: %.4f, NPV: %.4f, AUC: %.4f ,MCC: %.4f, F1: %.4f" \
          % (recall * 100, spec * 100, acc * 100, prec * 100, npv * 100, auc, mcc, f1 * 100))


if __name__ == '__main__':
    main()