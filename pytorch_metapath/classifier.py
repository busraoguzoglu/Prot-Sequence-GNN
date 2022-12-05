import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy
import pandas as pd
import pickle


from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score

from sklearn.utils import shuffle

from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, concatenate, add
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

def get_train_test():

    protein_interactions_train = pd.read_csv(
        'created_tables/setup5/seq_to_seq_train_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["protein_sequence1", "protein_sequence2", "label"]  # set our own names for the columns
    )
    protein_interactions_train = protein_interactions_train.iloc[1:, :]
    # There is some problem with this file, maybe need to create new one
    protein_interactions_train = protein_interactions_train[0:9229]

    protein_interactions_test = pd.read_csv(
        'created_tables/setup5/seq_to_seq_test_withlabels.csv',
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["protein_sequence1", "protein_sequence2", "label"]  # set our own names for the columns
    )
    protein_interactions_test = protein_interactions_test.iloc[1:, :]

    print('train:', protein_interactions_train)
    print('test:', protein_interactions_test)
    #protein_interactions_train = protein_interactions_train.astype(int)
    #protein_interactions_test = protein_interactions_test.astype(int)

    # Change ids with embeddings:
    train_data = pd.DataFrame(columns=['embedding', 'label'])
    test_data = pd.DataFrame(columns=['embedding', 'label'])

    # Get node embeddings:
    # Get df_hum and df_vir

    with open('created_tables/setup5/df_hum.pkl', 'rb') as f:
        df_hum = pickle.load(f)

    with open('created_tables/setup5/df_vir.pkl', 'rb') as f:
        df_vir = pickle.load(f)

    print(df_hum)

    for row_index, row in protein_interactions_train.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        label = row['label']

        #print(sequence1)

        # Get embedding of this id's from df_hum and df_vir
        hum_embed = df_hum.loc[(df_hum['id'] == sequence1)]['embedding']
        vir_embed = df_vir.loc[(df_vir['id'] == sequence2)]['embedding']

        sequence1emb = numpy.array(hum_embed)[0]
        sequence2emb = numpy.array(vir_embed)[0]

        sequences = numpy.concatenate((sequence1emb, sequence2emb), axis=0)

        new_row = {'embedding': sequences, 'label': label}
        train_data = train_data.append(new_row, ignore_index=True)

    print('train data:', train_data)

    for row_index, row in protein_interactions_test.iterrows():
        sequence1 = row['protein_sequence1']
        sequence2 = row['protein_sequence2']
        label = row['label']

        # Get embedding of this id's from df_hum and df_vir
        hum_embed = df_hum.loc[(df_hum['id'] == sequence1)]['embedding']
        vir_embed = df_vir.loc[(df_vir['id'] == sequence2)]['embedding']

        sequence1emb = numpy.array(hum_embed)[0]
        sequence2emb = numpy.array(vir_embed)[0]

        sequences = numpy.concatenate((sequence1emb, sequence2emb), axis=0)
        new_row = {'embedding': sequences, 'label': label}
        test_data = test_data.append(new_row, ignore_index=True)

    print('test data:', test_data)

    return train_data, test_data

def main():

    # Get train/test datasets for classifier
    train_data, test_data = get_train_test()

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
    test_Y = numpy.array(test_Y).astype(numpy.float32)

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
        model.add(Input(shape=(256,)))
        model.add(Dense(128, activation='relu'))
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

        print('y_pred_label',  y_pred_label)
        print('y_true', y_true)

        #y_pred_label = get_predictions(y_pred_label)

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
    recall = sum(recalls)/count
    spec = sum(specs)/count
    npv = sum(npvs)/count
    acc = sum(accs)/count
    prec = sum(precs)/count
    mcc = sum(mccs)/count
    auc = sum(aucs)/count
    f1 = sum(f1s)/count

    print("Sensitivity: %.4f, Specificity: %.4f, Accuracy: %.4f, PPV: %.4f, NPV: %.4f, AUC: %.4f ,MCC: %.4f, F1: %.4f" \
          % (recall * 100, spec * 100, acc * 100, prec * 100, npv * 100, auc, mcc, f1 * 100))



if __name__ == '__main__':
    main()