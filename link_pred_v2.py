import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from stellargraph import IndexedArray

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML

def main():

    protein_interactions_train = pd.read_csv(
        'created_tables/setup2/seq_to_seq_train.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )
    protein_interactions_train = protein_interactions_train.iloc[1:, :]

    protein_interactions_train_labels = pd.read_csv(
        'created_tables/setup2/seq_to_seq_train_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target", "weight"]  # set our own names for the columns
    )
    protein_interactions_train_labels = protein_interactions_train_labels.iloc[1:, :]

    protein_interactions_test = pd.read_csv(
        'created_tables/setup2/seq_to_seq_test.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target"]  # set our own names for the columns
    )
    protein_interactions_test = protein_interactions_test.iloc[1:, :]

    protein_interactions_test_labels = pd.read_csv(
        'created_tables/setup2/seq_to_seq_test_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["source", "target", "weight"]  # set our own names for the columns
    )
    protein_interactions_test_labels = protein_interactions_test_labels.iloc[1:, :]

    protein_interactions_train = protein_interactions_train.astype(int)
    protein_interactions_test = protein_interactions_test.astype(int)
    protein_interactions_train_labels = protein_interactions_train_labels.astype(int)
    protein_interactions_test_labels = protein_interactions_test_labels.astype(int)

    frames = [protein_interactions_train, protein_interactions_test]
    protein_interactions = pd.concat(frames, ignore_index=True)

    frames_train = [protein_interactions_train]
    protein_interactions_train = pd.concat(frames_train, ignore_index=True)

    frames_test = [protein_interactions_test]
    protein_interactions_test = pd.concat(frames_test, ignore_index=True)

    protein_word_content = pd.read_csv(
        'created_tables/setup2/seq_to_word.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    #protein_word_content = protein_word_content.iloc[1:, :]
    protein_word_content.drop(columns=protein_word_content.columns[0], axis=1, inplace=True)
    protein_word_content = protein_word_content.iloc[1:, :]

    for col in protein_word_content.columns:
        print(col)

    protein_word_content.columns = ['source', 'target']

    for col in protein_word_content.columns:
        print(col)

    #protein_word_content["source"] = protein_word_content["source"].astype(int)
    #protein_word_content["target"] = protein_word_content["target"].astype(int)
    protein_word_content = protein_word_content.astype(int)

    # Nodes (words-sequences)

    all_seq_id = pd.read_csv(
        'created_tables/setup2/all_seq_id.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    all_seq_id = all_seq_id.iloc[1:, :]
    all_seq_id.drop(columns=all_seq_id.columns[0], axis=1, inplace=True)
    all_seq_id.columns = ['index']
    all_seq_id.set_index("index", inplace=True)
    all_seq_id = all_seq_id.astype(int)

    indexes = []
    for i in range(2741):
        if i < 2741:
            indexes.append(i)
    sequence_nodes = IndexedArray(index=indexes)


    all_word_id = pd.read_csv(
        'created_tables/setup2/all_word_id.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
    )

    all_word_id = all_word_id.iloc[1:, :]
    all_word_id.drop(columns=all_word_id.columns[0], axis=1, inplace=True)
    all_word_id.columns = ['index']
    all_word_id.set_index("index", inplace=True)
    all_word_id = all_word_id.astype(int)

    indexes = []
    for i in range(7502):
        if i > 2740:
            indexes.append(i)
    word_nodes = IndexedArray(index=indexes)

    print("word nodes:", word_nodes)
    print("sequence nodes:", sequence_nodes)


    print('all_seq_id: ', all_seq_id)
    print('all_word_id: ', all_word_id)
    print('protein_interactions_train: ', protein_interactions_train)
    print('protein_word_content: ', protein_word_content)

    #protein_interactions_train_2 = protein_interactions_train[0:5]

    # edge indexes should be unique!
    # Create test edges:
    interaction_edges = pd.DataFrame(
        {
            "source": [1172, 5, 4000],
            "target": [2259, 25, 4786]
        }
    )
    contain_edges = pd.DataFrame(
        {
            "source": [1172, 5, 4000],
            "target": [5000, 6000, 7000]
        }
    )

    edges = [interaction_edges, contain_edges]
    all_edges = pd.concat(edges, ignore_index=True)

    edge_type = []

    for i in range(len(interaction_edges)):
        edge_type.append('i')
    for i in range(len(contain_edges)):
        edge_type.append('c')

    all_edges['type'] = edge_type
    print(all_edges)



    edges = {
        "interacts": interaction_edges,
        "contains": contain_edges
    }

    protein_train_graph = StellarGraph({"protein": sequence_nodes, "words": word_nodes}, edges = all_edges, edge_type_column="type")
    #protein_train_graph = StellarGraph({"protein": sequence_nodes, "words": word_nodes}, edges=edges)
    #protein_train_graph = StellarGraph({"protein": all_seq_id, "word": all_word_id}, edges=edges)
    #protein_train_graph = StellarGraph({"protein": all_seq_id, "word": all_word_id}, {"interacts": protein_interactions_train, "contains": protein_word_content})
    #protein_train_graph_2 = StellarGraph({"protein": all_seq_id, "word": all_word_id}, {"interacts": protein_interactions_train_2, "contains": protein_word_content})

    print(protein_train_graph.info())

    # ----------------------------------------------------------------------------------------------------

    # Training

#########################################################################################################

    # Create own G_train, G_test graphs:
    G_test = protein_train_graph
    edge_ids_test = []
    edge_labels_test = []
    for row_index, row in protein_interactions_test_labels.iterrows():
        node1 = row['target']
        node2 = row['source']
        label = row['weight']
        edge = [node1, node2]
        edge_ids_test.append(edge)
        edge_labels_test.append(label)

    edge_ids_test = np.array(edge_ids_test)
    edge_labels_test = np.array(edge_labels_test)
    print('own edge ids test:', edge_ids_test)
    print('own edge labels test:', edge_labels_test)

    G_train = protein_train_graph_2
    edge_ids_train = []
    edge_labels_train = []
    for row_index, row in protein_interactions_train_labels.iterrows():
        if row_index >= 5:
            node1 = row['target']
            node2 = row['source']
            label = row['weight']
            edge = [node1, node2]
            edge_ids_train.append(edge)
            edge_labels_train.append(label)

    edge_ids_train = np.array(edge_ids_train)
    edge_labels_train = np.array(edge_labels_train)
    print('own edge ids test:', edge_ids_train)
    print('own edge labels test:', edge_labels_train)


#########################################################################################################

    epochs = 100
    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

    gcn = GCN(
        layer_sizes=[512, 512], activations=["relu", "relu"], generator=train_gen, dropout=0.10
    )

    x_inp, x_out = gcn.in_out_tensors()

    prediction = LinkEmbedding(activation="relu", method="ip")(x_out)
    prediction = keras.layers.Reshape((-1,))(prediction)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.BinaryAccuracy()]
    )

    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(
        train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=True
    )

    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))


if __name__ == '__main__':
    main()

