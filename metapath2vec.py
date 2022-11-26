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

import matplotlib.pyplot as plt
import pickle

# %%
path = ('./')
proteins_path = osp.join(path, 'Proteins_bdb_all.csv')
chemicals_path = osp.join(path, 'Drugs_bdb_all2_removed_integer.csv')
protein_drug_path = osp.join(path, 'Drug_Protein_Edges_bdb_all_removed.csv')


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

class ProtEncoder(object):
    def __init__(self, model_name='Rostlab/prot_bert'):
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name)

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

class ChemEncoder(object):
    def __init__(self, model_name='seyonec/PubChem10M_SMILES_BPE_450k'):
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)
        self.chemical_tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name)

    @lru_cache(maxsize=1024)
    def get_chemberta_embedding(self, smiles: str):
        print(smiles)
        tokens = self.chemical_tokenizer(smiles, return_tensors='pt')
        print(tokens)
        output = self.model(**tokens)
        print(output)
        return output.last_hidden_state.detach().numpy().mean(axis=1)

    def vectorize_chemicals(self, chemicals: List[str]):
        return np.vstack([self.get_chemberta_embedding(chemical) for chemical in chemicals])

    @torch.no_grad()
    def __call__(self, df):
        x = self.vectorize_chemicals(df.values)
        return torch.from_numpy(x)

# protein_x, protein_mapping = load_node_csv(proteins_path,
#                                            index_col='uniprot_id',
#                                            encoders={
#                                                'sequence': ProtEncoder()}
#                                            )
#
# print(protein_x)
# print(protein_mapping)


# ligand_x, ligand_mapping = load_node_csv(chemicals_path,
#                                          index_col='pubchem_cid',
#                                          encoders={
#                                              'smiles': ChemEncoder()}
#                                          )
# %%
# ligand_x = pickle.load(otherfile)
# ligand_mapping = pickle.load(otherfile)
#

file = open('mappings_bdb_all.pk','rb')   #open('proteinMappings_bindingdb.pk','rb')
ligand_x = pickle.load(file)
ligand_mapping = pickle.load(file)
print(ligand_x)
print(ligand_mapping)

protein_x = pickle.load(file)
protein_mapping = pickle.load(file)
print(protein_x)
print(protein_mapping)

proteinWord_x = pickle.load(file)
proteinWord_mapping = pickle.load(file)
print(proteinWord_x)
print(proteinWord_mapping)

drugWord_x = pickle.load(file)
drugWord_mapping = pickle.load(file)
print(drugWord_x)
print(drugWord_mapping)


edge_index_protein, edge_label_protein = load_edge_csv(
    protein_drug_path,
    src_index_col='uniprot_id',
    src_mapping=protein_mapping,
    dst_index_col='pubchem_cid',
    dst_mapping=ligand_mapping
)

data = HeteroData()
data['ligand'].num_nodes = len(ligand_mapping)
data['protein'].x = protein_x

data['protein', 'protein_interacts_with_ligand', 'ligand'].edge_index = edge_index_protein
data['protein', 'protein_interacts_with_ligand', 'ligand'].edge_label = edge_label_protein

print(data)
data = ToUndirected()(data)

del data['ligand','rev_protein_interacts_with_ligand','protein'].edge_label_protein

transform = RandomLinkSplit(
    is_undirected=True,
    num_val=0.05,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('protein', 'protein_interacts_with_ligand', 'ligand')],
    rev_edge_types=[('ligand','rev_protein_interacts_with_ligand','protein')]
)
train_data, val_data, test_data = transform(data)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

metapath = [
    ('ligand', 'rev_protein_interacts_with_ligand', 'protein'),
    ('protein', 'protein_interacts_with_ligand', 'ligand')
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

def train(n_epochs, log_steps=5, eval_steps=10):
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

    return losses


losses = train(n_epochs=10)

def get_embedding(node_type, batch=None):
    emb = model.embedding.weight[model.start[node_type]:model.end[node_type]]
    return emb if batch is None else emb[batch]

ligand_embedding = get_embedding("ligand")
protein_embedding = get_embedding("protein")
