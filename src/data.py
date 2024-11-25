from typing import Dict, List, Optional, Union
import numpy as np
import anndata as ad
import pytorch_lightning as pl
import scanpy as sc
import scipy as sp
import torch
import pandas as pd
from torch.utils.data import DataLoader, IterableDataset

from src.dataclasses import SparseExpressionData


class CrossSpeciesDataset(IterableDataset):
    """Dataset for cross-species gene expression data."""

    def __init__(
        self,
        species_data: Dict[str, ad.AnnData],
        batch_size: int = 128,
        seed: int = 0,
    ):
        """
        Initialize dataset.

        Args:
            species_data: Dictionary mapping species names to preprocessed AnnData objects
            batch_size: Number of cells per batch (per species)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        
        # Store species names and create concatenated data
        self.species_names = list(species_data.keys())
        self.concatenated_data = self._concatenate_species_data(species_data)
        
        # Create species indices
        self.species_to_idx = {name: idx for idx, name in enumerate(self.species_names)}
        self.species_indices = {
            species: np.where(self.concatenated_data.obs["species"] == species)[0]
            for species in self.species_names
        }
        
        # Calculate sampling statistics
        self.n_cells_per_species = {
            species: len(indices) for species, indices in self.species_indices.items()
        }
        self.max_cells = max(self.n_cells_per_species.values())
        
        # Initialize sampling state
        self.rng = np.random.RandomState(seed)
        self._init_sampling_state()
        
        self.worker_id = None
        self.num_workers = None

    def _concatenate_species_data(self, species_data):
        """Concatenate data from different species in a block-diagonal fashion."""
        species_list = list(species_data.values())
        
        # Block concatenate of X data across species
        block_diag_X = sp.sparse.block_diag([data.X for data in species_list], format="csr")
        concat_obs = pd.concat([data.obs for data in species_list])
        concat_var = pd.concat([data.var for data in species_list])
        var_names = pd.Index(sum([list(data.var_names) for data in species_list], []))
        concat_var.index = var_names
        obs_names = pd.Index(sum([list(data.obs_names) for data in species_list], []))
        concat_obs.index = obs_names

        uns = {}
        for data in species_list:
            uns.update(data.uns)

        concat_adata = ad.AnnData(
            X=block_diag_X,
            obs=concat_obs,
            var=concat_var,
            uns=uns
        )
        return concat_adata

    def _init_sampling_state(self):
        """Initialize sampling state for each species."""
        self.available_indices = {
            species: indices.copy() 
            for species, indices in self.species_indices.items()
        }
        self.seen_largest_dataset = set()
        
        # Find species with max cells for epoch tracking
        self.largest_species = max(
            self.n_cells_per_species.items(), 
            key=lambda x: x[1]
        )[0]

    def _get_batch_indices(self):
        """Get balanced batch of indices across species."""
        batch_indices = []
        cells_per_species = self.batch_size // len(self.species_names)
        
        for species in self.species_names:
            available = self.available_indices[species]
            
            if len(available) >= cells_per_species:
                # Sample without replacement
                selected_idx = self.rng.choice(
                    available, 
                    cells_per_species, 
                    replace=False
                )
                self.available_indices[species] = np.setdiff1d(
                    available, 
                    selected_idx
                )
            else:
                selected_idx = available
                num_missing = cells_per_species - len(selected_idx)
                # Sample with replacement from original indices
                selected_idx = np.concatenate([
                    selected_idx,
                    self.rng.choice(
                        self.species_indices[species], 
                        num_missing, 
                        replace=True
                    )
                ])
            
            batch_indices.extend(selected_idx)
            
            # Track cells seen from largest dataset
            if species == self.largest_species:
                self.seen_largest_dataset.update([i for i in selected_idx])
        
        return np.array(batch_indices)

    def _epoch_finished(self):
        """Check if we've seen all cells from the largest dataset."""
        return len(self.seen_largest_dataset) >= self.n_cells_per_species[self.largest_species]

    def _create_sparse_batch(self, indices):
        """Create a SparseExpressionData batch from indices."""
        batch_data = self.concatenated_data[indices]
        batch_idx, gene_idx = batch_data.X.nonzero()
        values = torch.from_numpy(batch_data.X.data.astype(np.float32))
        gene_idx = torch.from_numpy(gene_idx.astype(np.int32))
        batch_idx = torch.from_numpy(batch_idx.astype(np.int32))
        
        species_idx = torch.tensor(
            [self.species_to_idx[s] for s in batch_data.obs["species"]],
            dtype=torch.int32
        )
        
        return SparseExpressionData(
            values=values,
            batch_idx=batch_idx,
            gene_idx=gene_idx,
            species_idx=species_idx,
            batch_size=len(indices),
            n_genes=len(self.concatenated_data.var_names),
            n_species=len(self.species_names),
        )

    def _initialize_worker_info(self):
        """Initialize worker information for distributed training."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
    
    def __len__(self):
        """Return the number of batches per epoch."""
        # We sample batch_size cells total, evenly distributed across species
        cells_per_species = self.batch_size // len(self.species_names)
        # Number of batches needed to see all cells from largest species once
        cells_in_largest = self.n_cells_per_species[self.largest_species]
        # Since we sample cells_per_species cells from largest dataset in each batch,
        # we need cells_in_largest / cells_per_species batches to see all cells once
        return int(np.ceil(cells_in_largest / cells_per_species))
    
    def __iter__(self):
        """Iterate over batches of data."""
        self._initialize_worker_info()
        self._init_sampling_state()
        
        while not self._epoch_finished():
            indices = self._get_batch_indices()
            yield self._create_sparse_batch(indices)


class CrossSpeciesDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for cross-species data."""

    def __init__(
        self,
        species_data: Dict[str, Union[str, List[str], ad.AnnData, List[ad.AnnData]]],
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 0,
    ):
        """
        Initialize data module.

        Args:
            species_data: Dictionary mapping species to data, where data can be:
                - A single AnnData object
                - A single file path (string) to h5ad file
                - List of AnnData objects
                - List of file paths to h5ad files
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.species_data = species_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Set random seed for reproducibility
        np.random.seed(seed)

    def _load_species_data(self, data):
        """Load and concatenate data for a single species."""
        if isinstance(data, (str, list)):
            # Load from file(s)
            if isinstance(data, str):
                data = [data]
            adatas = [ad.read_h5ad(path) if isinstance(path, str) else path for path in data]
            species_adata = ad.concat(adatas) if len(adatas) > 1 else adatas[0]
        else:
            # Direct AnnData object(s)
            if isinstance(data, list):
                species_adata = ad.concat(data) if len(data) > 1 else data[0]
            else:
                species_adata = data
        
        # Remove expression data from raw and layers and convert to sparse
        del species_adata.raw
        del species_adata.layers
        if not sp.sparse.issparse(species_adata.X):
            species_adata.X = sp.sparse.csr_matrix(species_adata.X)
            
        return species_adata

    def _split_species_data(self, adata, train_frac=0.8, val_frac=0.1, test_frac=0.1):
        """Split AnnData object into train/val/test sets."""
        n_cells = adata.n_obs
        indices = np.random.permutation(n_cells)
        
        train_size = int(train_frac * n_cells)
        val_size = int(val_frac * n_cells)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        return train_idx, val_idx, test_idx

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        # Load and split data for each species
        train_data = {}
        val_data = {}
        test_data = {}
        
        for species, data in self.species_data.items():
            # Load data for this species
            species_adata = self._load_species_data(data)
            species_adata.obs["species"] = species
            
            # Split the data
            train_frac = 1.0 - self.val_split - self.test_split
            train_idx, val_idx, test_idx = self._split_species_data(
                species_adata, 
                train_frac=train_frac,
                val_frac=self.val_split,
                test_frac=self.test_split
            )
            
            # Create split datasets
            train_data[species] = species_adata[train_idx].copy()
            val_data[species] = species_adata[val_idx].copy()
            test_data[species] = species_adata[test_idx].copy()
        
        if stage == "fit" or stage is None:
            self.train_dataset = CrossSpeciesDataset(
                species_data=train_data,
                batch_size=self.batch_size,
                seed=self.seed,
            )
            
            self.val_dataset = CrossSpeciesDataset(
                species_data=val_data,
                batch_size=self.batch_size,
                seed=self.seed,
            )

        if stage == "test" or stage is None:
            self.test_dataset = CrossSpeciesDataset(
                species_data=test_data,
                batch_size=self.batch_size,
                seed=self.seed,
            )

    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=self.num_workers,
        )

    @property
    def n_genes(self) -> int:
        """Total number of genes across all species."""
        if self.train_dataset is None:
            raise RuntimeError("DataModule not set up yet. Call setup() first.")
        return self.train_dataset.concatenated_data.shape[1]

    @property
    def n_species(self) -> int:
        """Number of species in the dataset."""
        if self.train_dataset is None:
            raise RuntimeError("DataModule not set up yet. Call setup() first.")
        return len(self.train_dataset.species_names)
