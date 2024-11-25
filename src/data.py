import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.dataclasses import SparseExpressionData


class CrossSpeciesDataset(IterableDataset):
    """Dataset for cross-species gene expression data."""

    def __init__(
        self,
        files_list: List[str],
        species_map: Dict[str, int],
        gene_vocab: Dict[str, int],
        data_dir: str = None,
        max_steps: int = 5000,
        batch_size: int = 128,
        gene_col_name: str = "feature_id",
        filter_to_vocab: bool = True,
        filter_outliers: float = 0.0,
        min_expressed_genes: int = 200,
        seed: int = 0,
        normalize_to_scale: bool = None,
        clip_counts: float = 1e10,
        log_norm_counts: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            files_list: List of AnnData file paths
            species_map: Mapping from species names to indices
            gene_vocab: Mapping from gene names to indices
            data_dir: Base directory for data files
            max_steps: Maximum number of steps per epoch
            batch_size: Batch size
            gene_col_name: Column name for gene IDs
            filter_to_vocab: Whether to filter genes to vocabulary
            filter_outliers: Standard deviation threshold for outlier filtering
            min_expressed_genes: Minimum number of expressed genes per cell
            seed: Random seed
            normalize_to_scale: Whether to normalize counts to a specific scale
            clip_counts: Maximum value for count clipping
            log_norm_counts: Whether to apply log normalization
        """
        super().__init__()
        self.files_list = files_list
        self.species_map = species_map
        self.gene_vocab = gene_vocab
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gene_col_name = gene_col_name
        self.filter_to_vocab = filter_to_vocab
        self.filter_outliers = filter_outliers
        self.min_expressed_genes = min_expressed_genes
        self.seed = seed
        self.normalize_to_scale = normalize_to_scale
        self.clip_counts = clip_counts
        self.log_norm_counts = log_norm_counts

        self.worker_id = None
        self.num_workers = None

    def _initialize_worker_info(self):
        """Initialize worker information for distributed training."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

            # Divide files among workers
            per_worker = int(np.ceil(len(self.files_list) / float(self.num_workers)))
            start_idx = self.worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.files_list))
            self.files_list = self.files_list[start_idx:end_idx]

    def _get_batch_from_file(self, file_path: str) -> Optional[SparseExpressionData]:
        """Get a batch of data from a file."""
        file_path = (
            os.path.join(self.data_dir, file_path) if self.data_dir else file_path
        )

        # Load data
        try:
            adata = sc.read_h5ad(file_path)
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")
            return None

        # Get gene names
        try:
            var_file = os.path.join(os.path.dirname(file_path), "var.csv")
            if os.path.exists(var_file):
                gene_names = pd.read_csv(var_file)[self.gene_col_name].values
            else:
                gene_names = adata.var[self.gene_col_name].values
        except:
            print(f"Failed to get gene names from {file_path}")
            return None

        # Convert to dense if sparse
        X = (
            adata.X.toarray()
            if isinstance(adata.X, (csr_matrix, csc_matrix))
            else np.array(adata.X)
        )
        if "free_annotation_v1" in adata.obs.columns:
            adata.obs["species"] = "Microcebus murinus"
        elif "organism" in adata.obs.columns:
            adata.obs["species"] = adata.obs["organism"]
        else:
            adata.obs["species"] = "Danio rerio"
        # Get species info
        species = adata.obs["species"].iloc[
            0
        ]  # All cells in a file are from same species
        species_idx = self.species_map[species]

        # Filter and normalize
        if self.filter_outliers > 0:
            gene_means = np.mean(X, axis=0)
            gene_stds = np.std(X, axis=0)
            outlier_mask = np.abs(X - gene_means) < self.filter_outliers * gene_stds
            X = X * outlier_mask

        if self.normalize_to_scale:
            sc.pp.normalize_total(adata, target_sum=1e4)

        if self.log_norm_counts:
            sc.pp.log1p(adata)

        if self.clip_counts < float("inf"):
            X = np.clip(X, 0, self.clip_counts)

        # Sample cells for batch
        n_cells = X.shape[0]
        batch_size = min(self.batch_size, n_cells)
        cell_indices = np.random.choice(n_cells, batch_size, replace=False)
        X_batch = X[cell_indices]

        # Convert to sparse format
        nonzero_mask = X_batch > 0
        values = []
        batch_idx = []
        gene_idx = []

        for i in range(batch_size):
            nonzero_genes = np.nonzero(nonzero_mask[i])[0]
            if len(nonzero_genes) >= self.min_expressed_genes:
                values.extend(X_batch[i, nonzero_genes])
                batch_idx.extend([i] * len(nonzero_genes))
                gene_idx.extend(
                    [self.gene_vocab.get(g, 0) for g in gene_names[nonzero_genes]]
                )

        if not values:  # No valid cells found
            return None

        # Convert to tensors
        return SparseExpressionData(
            values=torch.FloatTensor(values),
            batch_idx=torch.LongTensor(batch_idx),
            gene_idx=torch.LongTensor(gene_idx),
            species_idx=torch.LongTensor([species_idx] * batch_size),
            batch_size=batch_size,
            n_genes=len(self.gene_vocab),
        )

    def __iter__(self):
        """Iterator for the dataset."""
        # Initialize worker info if not already done
        if self.worker_id is None:
            self._initialize_worker_info()

        # Set random seed for reproducibility
        random.seed(
            self.seed + self.worker_id if self.worker_id is not None else self.seed
        )

        # Main iteration loop
        step = 0
        while step < self.max_steps:
            # Randomly select a file
            file_path = random.choice(self.files_list)

            # Get batch from file
            batch = self._get_batch_from_file(file_path)
            if batch is not None:
                yield batch
                step += 1


class CrossSpeciesDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for cross-species data."""

    def __init__(
        self,
        train_files: List[str],
        val_files: Optional[List[str]] = None,
        test_files: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        gene_vocab: Optional[Dict[str, int]] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        max_steps_per_epoch: int = 5000,
        gene_col_name: str = "feature_id",
        filter_to_vocab: bool = True,
        filter_outliers: float = 0.0,
        min_expressed_genes: int = 200,
        seed: int = 0,
        normalize_to_scale: bool = None,
        clip_counts: float = 1e10,
        log_norm_counts: bool = False,
        species_map: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize data module.

        Args:
            train_files: Training data files
            val_files: Validation data files (optional)
            test_files: Test data files (optional)
            data_dir: Base directory for data files
            gene_vocab: Optional pre-defined gene vocabulary
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            max_steps_per_epoch: Maximum steps per epoch
        """
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_steps = max_steps_per_epoch
        self.gene_col_name = gene_col_name
        self.filter_to_vocab = filter_to_vocab
        self.filter_outliers = filter_outliers
        self.min_expressed_genes = min_expressed_genes
        self.seed = seed
        self.normalize_to_scale = normalize_to_scale
        self.clip_counts = clip_counts
        self.log_norm_counts = log_norm_counts
        self.species_map = species_map
        self.gene_vocab = gene_vocab

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = CrossSpeciesDataset(
                files_list=self.train_files,
                species_map=self.species_map,
                gene_vocab=self.gene_vocab,
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                max_steps=self.max_steps,
                gene_col_name=self.gene_col_name,
                filter_to_vocab=self.filter_to_vocab,
                filter_outliers=self.filter_outliers,
                min_expressed_genes=self.min_expressed_genes,
                seed=self.seed,
                normalize_to_scale=self.normalize_to_scale,
                clip_counts=self.clip_counts,
                log_norm_counts=self.log_norm_counts,
            )

            if self.val_files is not None:
                self.val_dataset = CrossSpeciesDataset(
                    files_list=self.val_files,
                    species_map=self.species_map,
                    gene_vocab=self.gene_vocab,
                    data_dir=self.data_dir,
                    batch_size=self.batch_size,
                    max_steps=self.max_steps // 10,  # Fewer steps for validation
                    gene_col_name=self.gene_col_name,
                    filter_to_vocab=self.filter_to_vocab,
                    filter_outliers=self.filter_outliers,
                    min_expressed_genes=self.min_expressed_genes,
                    seed=self.seed,
                    normalize_to_scale=self.normalize_to_scale,
                    clip_counts=self.clip_counts,
                    log_norm_counts=self.log_norm_counts,
                )

        if stage == "test" or stage is None:
            if self.test_files is not None:
                self.test_dataset = CrossSpeciesDataset(
                    files_list=self.test_files,
                    species_map=self.species_map,
                    gene_vocab=self.gene_vocab,
                    data_dir=self.data_dir,
                    batch_size=self.batch_size,
                    max_steps=self.max_steps // 10,  # Fewer steps for testing
                    gene_col_name=self.gene_col_name,
                    filter_to_vocab=self.filter_to_vocab,
                    filter_outliers=self.filter_outliers,
                    min_expressed_genes=self.min_expressed_genes,
                    seed=self.seed,
                    normalize_to_scale=self.normalize_to_scale,
                    clip_counts=self.clip_counts,
                    log_norm_counts=self.log_norm_counts,
                )

    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Dataset already yields batches
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        if not hasattr(self, "val_dataset"):
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=None,  # Dataset already yields batches
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        if not hasattr(self, "test_dataset"):
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=None,  # Dataset already yields batches
            num_workers=self.num_workers,
            pin_memory=True,
        )
