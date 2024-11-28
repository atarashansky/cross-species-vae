from typing import Dict, Tuple
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
    lib_size_mean: torch.Tensor  # [batch_size]
    lib_size_logvar: torch.Tensor  # [batch_size]


@dataclass
class SpeciesLatents:
    mu: torch.Tensor  # Mean
    logvar: torch.Tensor  # Log variance
    latent: torch.Tensor  # Latent vector
    species_mask: torch.Tensor  # Species-specific mask

@dataclass
class EncoderOutput:
    """Output of encoder."""
    species_latents: Dict[int, SpeciesLatents]  # Species-specific latents
    global_latent: torch.Tensor  # Global latent
    global_mu: torch.Tensor
    global_logvar: torch.Tensor

@dataclass
class LossOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    homology_loss: torch.Tensor
    reconstruction: torch.Tensor