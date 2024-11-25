from dataclasses import dataclass

import torch

@dataclass
class SparseExpressionData:
    """Sparse representation of gene expression data."""

    values: torch.Tensor  # Expression values
    batch_idx: torch.Tensor  # Batch indices
    gene_idx: torch.Tensor  # Gene indices (using gene vocab tokens)
    species_idx: torch.Tensor  # Species indices
    batch_size: int  # Number of cells in batch
    n_genes: int  # Total number of genes in vocabulary
    n_species: int  # Total number of species