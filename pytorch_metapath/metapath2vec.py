import os.path as osp
import numpy as np
import re
import pandas as pd
import json
import pickle

from functools import lru_cache
from typing import List

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit

from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import MetaPath2Vec

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer

import matplotlib.pyplot as plt
import pickle


def load_node_csv(path, index_col, encoders=None, **kwargs):
    print('load_node_csv')
    #df = pd.read_csv(path, index_col=index_col, **kwargs)
    df = pd.read_csv(path,
                     sep="\t",  # tab-separated
                     header=None,  # no heading row
                     names=["id", "sequence"])
    print('read_csv:', df)
    df = df.iloc[1:, :]
    df.set_index(index_col, inplace=True)
    print('read_csv:', df)

    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    print('load_edge_csv')
    #df = pd.read_csv(path, **kwargs)

    df = pd.read_csv(path,
                     sep="\t",  # tab-separated
                     header=None,  # no heading row
                     names=["protein_sequence1", "protein_sequence2", "interaction"])

    df = df.iloc[1:, :]
    #df.set_index(index_col, inplace=True)

    print('read_csv')
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

class ProtEncoder(object):
    def __init__(self, model_name='prot_bert'):
        #self.protein_tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        #self.model = AutoModel.from_pretrained(model_name)
        self.protein_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name)

    print('BERT model initialized')

    @lru_cache(maxsize=1024)
    def get_protbert_embedding(self, aa_sequence: str):
        spaced_aa_sequence = " ".join(aa_sequence)
        print(aa_sequence)
        cleaned_sequence = re.sub(r'[UZOB]', 'X', spaced_aa_sequence)
        tokens = self.protein_tokenizer(cleaned_sequence, return_tensors='pt')
        print(tokens)
        output = self.model(**tokens)
        print(output)
        return output.last_hidden_state.detach().numpy().mean(axis=1)

    def vectorize_proteins(self, aa_sequences: List[str]):
        a = np.vstack([self.get_protbert_embedding(aa_sequence) for aa_sequence in aa_sequences])
        return a

    @torch.no_grad()
    def __call__(self, df):
        x = self.vectorize_proteins(df.values.tolist())
        return torch.from_numpy(x)

def train(model, loader, optimizer, device, n_epochs, log_steps=5, eval_steps=10):
    model.train()
    losses = []
    for epoch in range(n_epochs):
        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            print(loss.item())
            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                       f'Loss: {total_loss / log_steps:.4f}'))
                total_loss = 0
        losses.append(total_loss)

    return losses, model

def get_embedding(model, node_type, batch=None):
    emb = model.embedding.weight[model.start[node_type]:model.end[node_type]]
    return emb if batch is None else emb[batch]



def main():

    path = ('./')
    human_path = ('created_tables/setup5/all_seq_id_and_seq_human.csv')
    virus_path = ('created_tables/setup5/all_seq_id_and_seq_virus.csv')
    human_virus_edge_path = ('created_tables/setup5/training_edges_interact.csv')
    human_virus_edge_path2 = ('created_tables/setup5/training_edges_not_interact.csv')

    # chemicals_path = osp.join(path, 'Drugs_bdb_all2_removed_integer.csv')

    # protein_drug_path = osp.join(path, 'Drug_Protein_Edges_bdb_all_removed.csv')

    human_x, human_mapping = load_node_csv(human_path,
                                           index_col='id',
                                           encoders={
                                               'sequence': ProtEncoder()}
                                           )

    print('human_x:', human_x)
    print('human_mapping:', human_mapping)

    virus_x, virus_mapping = load_node_csv(virus_path,
                                           index_col='id',
                                           encoders={
                                               'sequence': ProtEncoder()}
                                           )

    print('virus_x:', virus_x)
    print('virus_mapping:', virus_mapping)

    edge_index_protein, edge_label_protein = load_edge_csv(
        human_virus_edge_path,
        src_index_col='protein_sequence1',
        src_mapping=human_mapping,
        dst_index_col='protein_sequence2',
        dst_mapping=virus_mapping
    )

    edge_index_protein2, edge_label_protein2 = load_edge_csv(
        human_virus_edge_path2,
        src_index_col='protein_sequence1',
        src_mapping=human_mapping,
        dst_index_col='protein_sequence2',
        dst_mapping=virus_mapping
    )

    print('Working till now')

    ###########################################################################################

    data = HeteroData()
    #data['ligand'].num_nodes = len(ligand_mapping)
    data['human'].x = human_x
    data['virus'].x = virus_x

    data['human', 'human_interacts_with_virus', 'virus'].edge_index = edge_index_protein
    data['human', 'human_interacts_with_virus', 'virus'].edge_label = edge_label_protein

    data['human', 'human_not_interacts_with_virus', 'virus'].edge_index = edge_index_protein2
    data['human', 'human_not_interacts_with_virus', 'virus'].edge_label = edge_label_protein2

    # Divide 'not interacts' and make them different file, add them to data.

    print(data)
    data = ToUndirected()(data)

    del data['human', 'human_interacts_with_virus', 'virus'].edge_label_protein
    del data['human', 'human_not_interacts_with_virus', 'virus'].edge_label_protein2

    transform = RandomLinkSplit(
        is_undirected=True,
        num_val=0.05,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('human', 'human_interacts_with_virus', 'virus'), ('human', 'human_not_interacts_with_virus', 'virus')],
        rev_edge_types=[('virus', 'rev_human_interacts_with_virus', 'human'), ('virus', 'rev_human_not_interacts_with_virus', 'human')]
    )
    train_data, val_data, test_data = transform(data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metapath = [
        ('human', 'human_interacts_with_virus', 'virus'),
        ('virus', 'rev_human_interacts_with_virus', 'human'),
        ('human', 'human_not_interacts_with_virus', 'virus'),
        ('virus', 'rev_human_not_interacts_with_virus', 'human')
    ]

    model = MetaPath2Vec(data.edge_index_dict,
                         embedding_dim=128,
                         metapath=metapath,
                         walk_length=10,
                         context_size=5,
                         walks_per_node=10,
                         num_negative_samples=5,
                         sparse=True
                         ).to(device)
    loader = model.loader(batch_size=32, shuffle=True)

    for idx, (pos_rw, neg_rw) in enumerate(loader):
        if idx == 10:
            break
        print(idx, pos_rw.shape, neg_rw.shape)

    print(pos_rw[0], neg_rw[0])

    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    losses, model = train(model, loader, optimizer, device, n_epochs=10)

    human_embedding = get_embedding(model, "human")
    virus_embedding = get_embedding(model, "virus")

    print('human_embedding:', human_embedding)
    print('virus_embedding:', virus_embedding)

    with open('human_embedding.pkl', 'wb') as handle:
        pickle.dump(human_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('virus_embedding.pkl', 'wb') as handle:
        pickle.dump(virus_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()



