import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import scipy as sp
import json
import numpy as np
from glob import glob
import anndata
import pandas as pd 
from src.vae import CrossSpeciesVAE
from src.data import CrossSpeciesDataModule

fn1 = '../samap/example_data/planarian.h5ad'
fn2 = '../samap/example_data/schistosome.h5ad'
fn3 = '../samap/example_data/hydra.h5ad'

eggnogs = '../samap/example_data/eggnog/*'

adata1 = anndata.read_h5ad(fn1)
adata2 = anndata.read_h5ad(fn2)
adata3 = anndata.read_h5ad(fn3)



dfs = []
for f in glob(eggnogs):
    dfs.append(pd.read_csv(f,sep='\t',header=None,skiprows=1))

df = pd.concat(dfs,axis=0)

ogs = set()
for i in df[18].values:
    for j in i.split(','):
        ogs.add(j)
ogs = list(ogs)

ogs_per_gene = [i.split(',') for i in df[18].values]
genes = df.iloc[:,0].values
ogs_per_gene = dict(zip(genes,ogs_per_gene))

indexer = pd.Series(index=ogs,data=range(len(ogs)))

x = []
y = []
z=[]
for i, g in enumerate(ogs_per_gene):
    v = indexer[ogs_per_gene[g]].values
    x.extend([i]*len(v))
    y.extend(v)
    z.extend(np.ones_like(v))
    
onehot = sp.sparse.coo_matrix((z,(x,y)),shape=(len(ogs_per_gene),len(indexer)))

graph = onehot.dot(onehot.T)

p1, p2 = graph.nonzero()

scores = graph.data
filt = scores > 2
homology_edges = torch.tensor(np.vstack((p1,p2)).T)[filt]
homology_scores = torch.tensor(scores)[filt]

# First, let the data module setup
data_module = CrossSpeciesDataModule(
    species_data = {
        "planarian": adata1,
        "schisto": adata2,
        "hydra": adata3,
    },
    batch_size=32,
    num_workers=0,
    val_split=0.1,
    test_split=0.1,
    seed=0
)
data_module.setup()

# Get species vocabulary sizes from data module
species_vocab_sizes = data_module.species_vocab_sizes

# Initialize the model using data module properties
model = CrossSpeciesVAE(
    species_vocab_sizes=species_vocab_sizes,
    homology_edges=homology_edges,
    homology_scores=homology_scores,
    n_latent=128,
    hidden_dims=[512, 256, 128],
    dropout_rate=0.1,
    learning_rate=1e-3,
    min_learning_rate=1e-5,
    warmup_epochs=0.1,
    init_beta=1.0,
    min_beta=0.0,
    max_beta=10.0,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)

# Initialize the trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=5,
    precision='16-mixed',
    gradient_clip_val=model.gradient_clip_val,
    gradient_clip_algorithm="norm",
    log_every_n_steps=10,
    deterministic=True,
    callbacks=[ModelCheckpoint(
        dirpath="checkpoints",
        filename="crossspecies_vae-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )],
    accumulate_grad_batches=len(species_vocab_sizes),  # Number of species
    enable_progress_bar=True,
    fast_dev_run=False,
)

trainer.fit(model, data_module)