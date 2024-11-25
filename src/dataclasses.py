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
    _index_mapping: dict | None = None

    def get_batch(self, idx: int) -> "SparseExpressionData":
        """Fast batch indexing using pre-computed mapping."""
        if not self._index_mapping:
            batch_mask = self.batch_idx == idx
            values = self.values[batch_mask]
            species_idx_batch = (
                self.species_idx[batch_mask][0]
                if len(self.species_idx[batch_mask]) > 0
                else self.species_idx[0]
            )
            return SparseExpressionData(
                values=values,
                batch_idx=torch.zeros(values.shape[0], dtype=self.batch_idx.dtype),
                gene_idx=self.gene_idx[batch_mask],
                species_idx=species_idx_batch.unsqueeze(0),
                batch_size=1,
                n_genes=self.n_genes,
            )

        indices = self._index_mapping.get(idx, [])
        species_idx_batch = (
            self.species_idx[indices][0] if len(indices) > 0 else self.species_idx[0]
        )
        return SparseExpressionData(
            values=self.values[indices],
            batch_idx=torch.zeros(len(indices), dtype=self.batch_idx.dtype),
            gene_idx=self.gene_idx[indices],
            species_idx=species_idx_batch.unsqueeze(0),
            batch_size=1,
            n_genes=self.n_genes,
        )
