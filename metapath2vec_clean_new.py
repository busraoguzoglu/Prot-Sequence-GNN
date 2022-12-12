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
    print('preds:', y_pred_value[0])
    print('preds:', len(y_pred_value))
    print('preds:', len(y_pred_value[0]))

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
        'created_tables/clean_metapath/seq_to_word_train_tfidf_filtered.csv',
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
        'created_tables/clean_metapath/seq_to_seq_train_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target", "interaction_type"]  # set our own names for the columns
    )

    # Need to make all edges have unique id
    train_labels_index = []
    for i in range(9753):
        train_labels_index.append(i + len(protein_word_content) + 1)

    protein_interactions_train_labels = protein_interactions_train_labels.iloc[1:, :]
    protein_interactions_train_labels['index'] = train_labels_index
    protein_interactions_train_labels.set_index(['index'], inplace=True)
    protein_interactions_train_labels = protein_interactions_train_labels.astype(int)

    print('protein_interactions_train_labels (interaction edges):', protein_interactions_train_labels)

    interacts = protein_interactions_train_labels[protein_interactions_train_labels['interaction_type'] == 1]
    not_interacts = protein_interactions_train_labels[protein_interactions_train_labels['interaction_type'] == 0]

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
    # Similarity Edges #
    #################################################################################
    # Add human-human virus-virus similarity edges:
    hum_similarity_edges_lev = pd.read_csv(
        'created_tables/clean_metapath/lev_dist_edges_hum.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )
    hum_similarity_edges_lev = hum_similarity_edges_lev.iloc[1:, :]

    # Need to make all edges have unique id
    hum_similarity_edges_index = []
    edge_start = word_word_edges_index[len(word_word_edges_index)-1]

    for i in range(len(hum_similarity_edges_lev)):
        hum_similarity_edges_index.append(i + edge_start + 1)

    hum_similarity_edges_lev['index'] = hum_similarity_edges_index
    hum_similarity_edges_lev.set_index(['index'], inplace=True)
    hum_similarity_edges_lev = hum_similarity_edges_lev.astype(int)
    print('hum_similarity_edges: (Levenstein)', hum_similarity_edges_lev)

    vir_similarity_edges_lev = pd.read_csv(
        'created_tables/clean_metapath/lev_dist_edges_vir.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )
    vir_similarity_edges_lev = vir_similarity_edges_lev.iloc[1:, :]

    # Need to make all edges have unique id
    vir_similarity_edges_index = []
    edge_start = hum_similarity_edges_index[len(hum_similarity_edges_index)-1]

    for i in range(len(vir_similarity_edges_lev)):
        vir_similarity_edges_index.append(i + edge_start + 1)

    vir_similarity_edges_lev['index'] = vir_similarity_edges_index
    vir_similarity_edges_lev.set_index(['index'], inplace=True)
    vir_similarity_edges_lev = vir_similarity_edges_lev.astype(int)
    print('vir_similarity_edges: (Levenstein)', vir_similarity_edges_lev)


    #################################################################################
    # Jaccard Edges #
    #################################################################################
    # Add human-human virus-virus similarity edges:
    hum_similarity_edges_jac = pd.read_csv(
        'created_tables/clean_metapath/jaccard_sim_edges_hum.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )
    hum_similarity_edges_jac = hum_similarity_edges_jac.iloc[1:, :]

    # Need to make all edges have unique id
    hum_similarity_edges_index = []
    edge_start = vir_similarity_edges_index[len(vir_similarity_edges_index) - 1]

    for i in range(len(hum_similarity_edges_jac)):
        hum_similarity_edges_index.append(i + edge_start + 1)

    hum_similarity_edges_jac['index'] = hum_similarity_edges_index
    hum_similarity_edges_jac.set_index(['index'], inplace=True)
    hum_similarity_edges_jac = hum_similarity_edges_jac.astype(int)
    print('hum_similarity_edges (Jaccard):', hum_similarity_edges_jac)

    vir_similarity_edges_jac = pd.read_csv(
        'created_tables/clean_metapath/jaccard_sim_edges_vir.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )
    vir_similarity_edges_jac = vir_similarity_edges_jac.iloc[1:, :]

    # Need to make all edges have unique id
    vir_similarity_edges_index = []
    edge_start = hum_similarity_edges_index[len(hum_similarity_edges_index) - 1]

    for i in range(len(vir_similarity_edges_jac)):
        vir_similarity_edges_index.append(i + edge_start + 1)

    vir_similarity_edges_jac['index'] = vir_similarity_edges_index
    vir_similarity_edges_jac.set_index(['index'], inplace=True)
    vir_similarity_edges_jac = vir_similarity_edges_jac.astype(int)
    print('hum_similarity_edges (Jaccard):', vir_similarity_edges_jac)


    #########################################################################################
    ### NODES ####
    #########################################################################################

    # Get human, virus and word indexes:
    hum_ids = pd.read_csv(
        'created_tables/clean_metapath/all_seq_id_human_train.csv',
        sep="\t",  # tab-separated
    )
    hum = hum_ids['id'].tolist()

    vir_ids = pd.read_csv(
        'created_tables/clean_metapath/all_seq_id_virus_train.csv',
        sep="\t",  # tab-separated
    )
    vir = vir_ids['id'].tolist()

    word_ids = pd.read_csv(
        'created_tables/clean_metapath/all_word_id_train.csv',
        sep="\t",  # tab-separated
    )
    word = word_ids['id'].tolist()

    # Create nodes (hum-vir-word)
    hum_nodes = IndexedArray(index=hum)
    vir_nodes = IndexedArray(index=vir)
    word_nodes = IndexedArray(index=word)

    protein_graph_train = StellarGraph({"human": hum_nodes, "virus": vir_nodes, "word": word_nodes},
                                 edges={'contains': protein_word_content, 'interacts': interacts, 'not_interacts': not_interacts, 'pmi': word_word_edges,
                                        'hum_similarity_lev': hum_similarity_edges_lev, 'vir_similarity_lev': vir_similarity_edges_lev,
                                        'hum_similarity_jac': hum_similarity_edges_jac, 'vir_similarity_jac': vir_similarity_edges_jac})


    print(protein_graph_train.info())
    return protein_graph_train

def get_train_node_embeddings(g):

    walk_length = 250  # maximum length of a random walk to use throughout this notebook

    # specify the metapath schemas as a list of lists of node types.
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

    print('train interactions file:', protein_interactions_train)

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

    # Change ids with embeddings:
    # Go over test data, calculate similarities, and calculate embeddings:
    test_data = pd.DataFrame(columns=['embedding', 'label'])

    for row_index, row in protein_interactions_test.iterrows():
        sequence1id = row['protein_sequence1']
        sequence2id = row['protein_sequence2']
        label = row['label']

        # Find test sequences:
        row1 = seq_to_id_test.iloc[sequence1id]
        sequence1 = row1['sequence']
        row2 = seq_to_id_test.iloc[sequence2id]
        sequence2 = row2['sequence']

        sequence1_similarity_values = []
        sequence2_similarity_values = []
        sequence1_similarity_trainseqid = []
        sequence2_similarity_trainseqid = []

        #########################################################################

        # For each sequence, find the most similar 5 train sequences:
        for row_index, row in seq_to_id_train.iterrows():
            id = row_index
            train_sequence = row['sequence']
            # Calculate levenstein distance (can maybe change to jaccard later)
            lev1 = distance(train_sequence, sequence1)
            lev2 = distance(train_sequence, sequence2)
            sequence1_similarity_values.append(lev1)
            sequence2_similarity_values.append(lev2)
            sequence1_similarity_trainseqid.append(id)
            sequence2_similarity_trainseqid.append(id)

        # Sort similarity values, get most similar 5 ids for seq1 and seq2
        sequence1_similarity_values, sequence1_similarity_trainseqid = zip(
            *sorted(zip(sequence1_similarity_values, sequence1_similarity_trainseqid)))
        sequence2_similarity_values, sequence2_similarity_trainseqid = zip(
            *sorted(zip(sequence2_similarity_values, sequence2_similarity_trainseqid)))

        sequence1_similarity_trainseqid = sequence1_similarity_trainseqid[0:3]
        sequence2_similarity_trainseqid = sequence2_similarity_trainseqid[0:3]

        # Now get the embedding of these five sequences for each,
        # and get the average. This will give embedding of the test sequence:

        ## Continue from here.
        # For sequence1:
        ar1 = node_embeddings_train[sequence1_similarity_trainseqid[0]]
        ar2 = node_embeddings_train[sequence1_similarity_trainseqid[1]]
        ar3 = node_embeddings_train[sequence1_similarity_trainseqid[2]]
        #ar4 = node_embeddings_train[sequence1_similarity_trainseqid[3]]
        #ar5 = node_embeddings_train[sequence1_similarity_trainseqid[4]]

        #seq1emb = (ar1+ar2+ar3+ar4+ar5)/5
        #seq1emb = ((2*ar1) + ar2) / 3

        seq1emb = ar1

        # For sequence2:
        ar1 = node_embeddings_train[sequence2_similarity_trainseqid[0]]
        ar2 = node_embeddings_train[sequence2_similarity_trainseqid[1]]
        ar3 = node_embeddings_train[sequence2_similarity_trainseqid[2]]
        #ar4 = node_embeddings_train[sequence2_similarity_trainseqid[3]]
        #ar5 = node_embeddings_train[sequence2_similarity_trainseqid[4]]

        #seq2emb = (ar1 + ar2 + ar3 + ar4 + ar5) / 5
        #seq2emb = ((2*ar1) + ar2) / 3

        seq2emb = ar1

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

    #######################################################
    #### Training ####
    #######################################################

    # Training the classifier #
    train_X = train_data[['embedding']]
    test_X = test_data[['embedding']]
    train_labels = train_data[['label']]
    test_labels = test_data[['label']]

    print(train_labels)
    print(test_labels)

    train_X, train_Y = shuffle(train_X, train_labels, random_state=0)
    test_X, test_Y = shuffle(test_X, test_labels, random_state=0)

    print(train_X)

    train_X_list = []
    for row_index, row in train_X.iterrows():
        emb = row['embedding']
        train_X_list.append(emb)

    test_X_list = []
    for row_index, row in test_X.iterrows():
        emb = row['embedding']
        test_X_list.append(emb)

    train_Y_list = []
    for row_index, row in train_Y.iterrows():
        lab = row['label']
        train_Y_list.append(lab)

    test_Y_list = []
    for row_index, row in test_Y.iterrows():
        lab = row['label']
        test_Y_list.append(lab)

    train_X = train_X_list
    test_X = test_X_list
    train_Y = train_Y_list
    test_Y = test_Y_list

    print('Shuffle done')

    print(len(train_X))
    print(train_X[0])
    print('shape of an entry:', train_X[0].shape)

    # Classification:

    # 2- Tensorflow:
    test_X = numpy.array(test_X)
    test_Y = numpy.array(test_Y)

    train_X = numpy.asarray(train_X).astype(numpy.float32)
    train_Y = numpy.asarray(train_Y).astype(numpy.float32)
    print(train_X.shape)

    epochs = 300
    batch_size = 128

    recalls, specs, npvs, accs, precs, mccs, aucs, f1s = [], [], [], [], [], [], [], []

    print('train_X shape:', train_X.shape)

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

        # y_true = val_Y
        # y_pred_label = model.predict(val_X)
        y_true = test_Y
        y_pred_label = model.predict(test_X)
        y_pred_label = get_predictions(y_pred_label)

        y_pred_label = list(y_pred_label)
        y_true = list(y_true)

        print('y_pred_label', y_pred_label)
        print('y_true', y_true)

        # y_pred_label = get_predictions(y_pred_label)

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