import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch
import scipy as sp
import json
import numpy as np
from glob import glob
import anndata
import pandas as pd
# from src.multi_vae_pemb import CrossSpeciesVAE
from src.multi_vae import CrossSpeciesVAE
from src.callbacks import StageAwareEarlyStopping
from src.data import CrossSpeciesDataModule
import pickle
from sklearn.metrics import adjusted_mutual_info_score
import scanpy as sc
import matplotlib.pyplot as plt
import umap
import hnswlib

def _tanh_scale(x,scale=10,center=0.5):
    return center + (1-center) * np.tanh(scale * (x - center))

def _united_proj(wpca1, wpca2, k=20, metric="cosine", ef=200, M=48):

    metric = 'l2' if metric == 'euclidean' else metric
    metric = 'cosine' if metric == 'correlation' else metric
    labels2 = np.arange(wpca2.shape[0])
    p2 = hnswlib.Index(space=metric, dim=wpca2.shape[1])
    p2.init_index(max_elements=wpca2.shape[0], ef_construction=ef, M=M)
    p2.add_items(wpca2, labels2)
    p2.set_ef(ef)
    idx1, dist1 = p2.knn_query(wpca1, k=k)

    if metric == 'cosine':
        dist1 = 1 - dist1
        dist1[dist1 < 1e-3] = 1e-3
        dist1 = dist1/dist1.max(1)[:,None]
        dist1 = _tanh_scale(dist1,scale=10, center=0.7)
    else:
        sigma1 = dist1[:,4]
        sigma1[sigma1<1e-3]=1e-3
        dist1 = np.exp(-dist1/sigma1[:,None])
        
    Sim1 = dist1  # np.exp(-1*(1-dist1)**2)
    knn1v2 = sp.sparse.lil_matrix((wpca1.shape[0], wpca2.shape[0]))
    x1 = np.tile(np.arange(idx1.shape[0])[:, None], (1, idx1.shape[1])).flatten()
    knn1v2[x1.astype('int32'), idx1.flatten().astype('int32')] = Sim1.flatten()
    return knn1v2.tocsr()

from pynndescent import NNDescent

def find_nearest_neighbors(L1, L2, n_neighbors=15, metric='correlation'):
    """
    Finds the nearest neighbors from L1 (query) to L2 (index) using pynndescent.

    Parameters:
        L1 (np.ndarray): Query embeddings of shape (num_queries, embedding_dim).
        L2 (np.ndarray): Index embeddings of shape (num_index, embedding_dim).
        n_neighbors (int): Number of neighbors to find. Default is 5.
        metric (str): Distance metric to use. Default is 'euclidean'.

    Returns:
        indices (np.ndarray): Indices of nearest neighbors in L2 for each query in L1.
        distances (np.ndarray): Distances to nearest neighbors for each query in L1.
    """
    # Validate inputs
    if not isinstance(L1, np.ndarray) or not isinstance(L2, np.ndarray):
        raise ValueError("L1 and L2 must be numpy arrays.")
    
    if L1.shape[1] != L2.shape[1]:
        raise ValueError("L1 and L2 must have the same embedding dimension.")

    # Build the index on L2
    index = NNDescent(L2, metric=metric, n_neighbors=n_neighbors)
    
    # Query the nearest neighbors for L1
    indices, distances = index.query(L1, k=n_neighbors)
    
    return indices, distances

adata1 = anndata.read_h5ad('data/wagner/data.h5ad')
adata2 = anndata.read_h5ad('data/briggs/data.h5ad')

adata1.X = adata1.X.astype('float32')
adata2.X = adata2.X.astype('float32')

emb1 = pickle.load(open('data/wagner/gene_embeddings.pkl','rb'))
emb2 = pickle.load(open('data/briggs/gene_embeddings.pkl','rb'))

emb1 = torch.cat([torch.tensor(emb1[i]).unsqueeze(-2) for i in adata1.var_names],dim=0).float()
emb2 = torch.cat([torch.tensor(emb2[i]).unsqueeze(-2) for i in adata2.var_names],dim=0).float()

XY_raw = _united_proj(emb1.numpy(), emb2.numpy(), k=25, metric='euclidean')
YX_raw = _united_proj(emb2.numpy(), emb1.numpy(), k=25, metric='euclidean')

XY = XY_raw.copy()
YX = YX_raw.copy()
XY.data[:]=1
YX.data[:]=1

G = XY + YX.T

G.data[G.data>1]=0
G.eliminate_zeros()
x, y = G.nonzero()


G = XY_raw/2 + YX_raw.T/2
G[x,y] = 0
G.eliminate_zeros()
x, y = G.nonzero()

homology_edges = {}
homology_edges[0] = {}
homology_edges[0][1] = torch.tensor(np.vstack((x,y)).T)

homology_edges[1] = {}
homology_edges[1][0] = torch.tensor(np.vstack((y,x)).T)

homology_scores = {}
homology_scores[0] = {}
homology_scores[0][1] = torch.tensor(G.data).float()

homology_scores[1] = {}
homology_scores[1][0] = torch.tensor(G.data).float()

species_data = {
    "wagner": adata1,
    "briggs": adata2,
}

emb_data = {
    "wagner": emb1,
    "briggs": emb2, 
}
data_module = CrossSpeciesDataModule(
    species_data = species_data,
    batch_size=512,
    num_workers=0,
    val_split=0.001,
    test_split=0.001,
    yield_pairwise=False,
    subsample_size=10000,
    subsample_by={
        "wagner": "cell_type",
        "briggs": "cell_type",   
    }
)
data_module.setup()

species_data_sub = {k: data_module.train_dataset.epoch_data[k][data_module.train_dataset.epoch_indices[k]].copy() for k in data_module.train_dataset.epoch_data}

batch_size = 512

data_module = CrossSpeciesDataModule(
    species_data = species_data_sub,
    batch_size=batch_size,
    num_workers=0,
    val_split=0.1,
    test_split=0.1,
    yield_pairwise=False,
    labels={
        "wagner": "cell_type",
        "briggs": "cell_type",
    }
)
data_module.setup()

emb_data = {data_module.train_dataset.species_to_idx[k]: v for k,v in emb_data.items()}

species_vocab_sizes = data_module.species_vocab_sizes

# Initialize the model using data module properties
model = CrossSpeciesVAE(
    species_vocab_sizes=species_vocab_sizes,
    homology_edges=homology_edges,
    homology_scores=homology_scores,
    batch_size=batch_size,
    direct_recon_weight=1.0,
    cross_species_recon_weight=1.0,

    triplet_epoch_start=0.5,
    triplet_loss_margin=0.2,
    
    transform_weight=0.0, #from 0.1
    base_learning_rate=1e-2,
    min_learning_rate=1e-4,
    
    warmup_data=0.1,
    homology_score_momentum=0.9,
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0,
    patience=11,
    verbose=True,
    mode='min'
)

# Initialize the trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=4,
    precision='16-mixed',
    gradient_clip_val=model.gradient_clip_val,
    gradient_clip_algorithm="norm",
    log_every_n_steps=1,
    deterministic=True,
    callbacks=[],
    accumulate_grad_batches=1,
    enable_progress_bar=True,
    fast_dev_run=False,
    logger=CSVLogger(
        save_dir="logs",
        name="metrics",
        flush_logs_every_n_steps=10
    )    
)

trainer.fit(model, data_module)
print(trainer.current_epoch)