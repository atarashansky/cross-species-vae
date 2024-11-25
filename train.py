import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import scipy as sp
import json
import numpy as np
from src.vae import ScalableCrossSpeciesVAE
from src.data import CrossSpeciesDataModule

if __name__ == "__main__":
    vocab = json.load(
        open("data/vocab/human_mouse_zfish_mlemur/protein_emb/gene_vocab.json", "r")
    )
    gene_orthology = json.load(
        open("data/vocab/orthology_groups/gene_orthology_group_vocab.json", "r")
    )
    size = gene_orthology["__SIZE__"]
    gene_orthology = {k: gene_orthology[k] for k in vocab if k in gene_orthology}

    # Hardcoded parameters
    params = {
        "n_latent": 128,  # Latent dimension
        "hidden_dims": [128, 256, 512],  # Hidden dimensions for encoder/decoder
        "dropout_rate": 0.1,
        "learning_rate": 1e-3,
        "species_dim": 32,  # Species embedding dimension
        "l1_lambda": 0.01,  # L1 regularization strength
        "temperature": 0.1,  # Temperature for Gumbel-Softmax
        # Training parameters
        "batch_size": 16,
        "max_steps": 100000,
        "max_epochs": 1000,
        "max_steps_per_epoch": 500,
        "gradient_accumulation_steps": 1,
        "gradient_clip_val": 1.0,
        # Distributed training
        "num_nodes": 1,
        "num_gpus_per_node": 1,
        # Data parameters
        "data_dir": "/mnt/czi-sci-ai/generate-cross-species-secondary/datasets",  # Update this path
        "train_files": "data/train_files/train_human_mouse_zfish_mlemur.txt",
        "val_files": "data/train_files/val_human_mouse_zfish_mlemur.txt",
    }

    train_files = open(params["train_files"]).read().split("\n")[:-1]
    val_files = open(params["val_files"]).read().split("\n")[:-1]

    # Set up data module
    data_module = CrossSpeciesDataModule(
        train_files=train_files,
        val_files=val_files,
        data_dir=params["data_dir"],
        batch_size=params["batch_size"],
        num_workers=1,
        max_steps_per_epoch=params["max_steps_per_epoch"],
        gene_vocab=vocab,
        gene_col_name="ensembl_id",
        species_map={
            "Homo sapiens": 0,
            "Mus musculus": 1,
            "Microcebus murinus": 2,
            "Danio rerio": 3,
        },
    )

    x = []
    y = []
    z = []
    for k in gene_orthology:
        v = gene_orthology[k]
        if not isinstance(v, list):
            continue
        x.extend([vocab[k]] * len(v))
        y.extend(v)
        z.extend([1] * len(v))

    G = sp.sparse.coo_matrix((z, (x, y)), shape=(len(vocab), size))
    G = G.dot(G.T)
    p1, p2 = G.nonzero()
    scores = G.data
    filt = scores > 2
    homology_edges = torch.tensor(np.vstack((p1, p2)).T)[filt]
    homology_scores = torch.tensor(scores)[filt]

    # Initialize the model
    model = ScalableCrossSpeciesVAE(
        n_genes=len(data_module.gene_vocab),
        n_species=len(data_module.species_map),
        n_latent=params["n_latent"],
        hidden_dims=params["hidden_dims"],
        dropout_rate=params["dropout_rate"],
        species_dim=params["species_dim"],
        l1_lambda=params["l1_lambda"],
        learning_rate=params["learning_rate"],
        num_nodes=params["num_nodes"],
        num_gpus_per_node=params["num_gpus_per_node"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        batch_size=params["batch_size"],
        max_steps=params["max_steps"],
        max_epochs=params["max_epochs"],
        max_steps_per_epoch=params["max_steps_per_epoch"],
        temperature=params["temperature"],
        gradient_clip_val=params["gradient_clip_val"],
        homology_edges=homology_edges,
        homology_scores=homology_scores,
    )
    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=params["num_gpus_per_node"],
        num_nodes=params["num_nodes"],
        max_epochs=params["max_epochs"],
        precision="16-mixed",
        accumulate_grad_batches=params["gradient_accumulation_steps"],
        gradient_clip_val=params["gradient_clip_val"],
        log_every_n_steps=10,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="scalable_vae-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            )
        ],
        enable_progress_bar=True,
        # fast_dev_run=True,
    )

    # Train the model
    trainer.fit(model, data_module)