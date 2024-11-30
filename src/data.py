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
        
        # Store species data directly
        self.species_data = species_data
        self.species_names = list(species_data.keys())
        
        # Create species indices
        self.species_to_idx = {name: idx for idx, name in enumerate(self.species_names)}
        
        # Create indices for each species
        self.species_indices = {
            species: np.arange(data.n_obs)
            for species, data in species_data.items()
        }
        
        # Initialize sampling state
        self.rng = np.random.RandomState(seed)
        self._init_sampling_state()
        
        self.worker_id = None
        self.num_workers = None


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
        """Get batch of indices for current species."""
        # Instead of balanced sampling, sample from one species
        species = self.current_species
        available = self.available_indices[species]
        
        if len(available) >= self.batch_size:
            selected_idx = self.rng.choice(
                available, 
                self.batch_size, 
                replace=False
            )
            self.available_indices[species] = np.setdiff1d(
                available, 
                selected_idx
            )
        else:
            selected_idx = available
            num_missing = self.batch_size - len(selected_idx)
            selected_idx = np.concatenate([
                selected_idx,
                self.rng.choice(
                    self.species_indices[species], 
                    num_missing, 
                    replace=True
                )
            ])
            
        return np.array(selected_idx)

    def _epoch_finished(self):
        """Check if we've seen all cells from the largest dataset."""
        return len(self.seen_largest_dataset) >= self.n_cells_per_species[self.largest_species]

    def _create_sparse_batch(self, indices):
        """Create a SparseExpressionData batch from indices."""
        species = self.current_species
        species_adata = self.species_data[species]
        
        # Get batch data
        batch_data = species_adata[indices]
        batch_idx, gene_idx = batch_data.X.nonzero()
        values = torch.from_numpy(batch_data.X.data.astype(np.float32))
        gene_idx = torch.from_numpy(gene_idx.astype(np.int32))
        batch_idx = torch.from_numpy(batch_idx.astype(np.int32))
        

        species_idx = torch.full(
            (len(indices),), 
            self.species_to_idx[species], 
            dtype=torch.int32
        )
        
        return SparseExpressionData(
            values=values,
            batch_idx=batch_idx,
            gene_idx=gene_idx,
            species_idx=species_idx,
            batch_size=len(indices),
            n_genes=species_adata.n_vars,
            n_species=len(self.species_names),
        )

    def _initialize_worker_info(self):
        """Initialize worker information for distributed training."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
    
    @property
    def n_cells_per_species(self):
        """Number of cells per species."""
        return {
            species: len(indices)
            for species, indices in self.species_indices.items()
        }
    
    def __len__(self):
        """Return the number of batches per epoch."""
        # Since we cycle through all species in round-robin fashion,
        # multiply the number of batches by number of species
        cells_in_largest = max(self.n_cells_per_species.values())
        batches_for_largest = int(np.ceil(cells_in_largest / self.batch_size))
        return batches_for_largest * len(self.species_names)
    
    def __iter__(self):
        """Iterate over species in round-robin fashion."""
        self._initialize_worker_info()
        self._init_sampling_state()
        
        while not self._epoch_finished():
            # Cycle through species
            for species in self.species_names:
                self.current_species = species
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
        multi_species_mode: bool = False,
        n_species_per_batch: int = 2,
        cluster_key: str | None = None,
        neighbors_per_cell: int = 100,
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
            multi_species_mode: Flag to enable multi-species mode
            n_species_per_batch: Number of species per batch in multi-species mode
            cluster_key: Key for cluster assignments in multi-species mode
            neighbors_per_cell: Number of neighbors per cell for coverage calculation
        """
        super().__init__()
        self.species_data = species_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        # Multi-species mode parameters
        self.multi_species_mode = multi_species_mode
        self.n_species_per_batch = n_species_per_batch
        self.cluster_key = cluster_key
        self.neighbors_per_cell = neighbors_per_cell
        
        if multi_species_mode and cluster_key is None:
            raise ValueError("cluster_key must be provided for multi_species_mode")

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
        
        # Choose dataset class based on mode
        if self.multi_species_mode:
            dataset_class = MultiSpeciesDataset
            dataset_kwargs = {
                'cluster_key': self.cluster_key,
                'n_species_per_batch': self.n_species_per_batch,
                'neighbors_per_cell': self.neighbors_per_cell,
            }
        else:
            dataset_class = CrossSpeciesDataset
            dataset_kwargs = {}
        
        if stage == "fit" or stage is None:
            self.train_dataset = dataset_class(
                species_data=train_data,
                batch_size=self.batch_size,
                seed=self.seed,
                **dataset_kwargs
            )
            
            self.val_dataset = dataset_class(
                species_data=val_data,
                batch_size=self.batch_size,
                seed=self.seed,
                **dataset_kwargs
            )

        if stage == "test" or stage is None:
            self.test_dataset = dataset_class(
                species_data=test_data,
                batch_size=self.batch_size,
                seed=self.seed,
                **dataset_kwargs
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

    @property
    def species_vocab_sizes(self) -> Dict[int, int]:
        """Get vocabulary sizes for each species."""
        if self.train_dataset is None:
            raise RuntimeError("DataModule not set up yet. Call setup() first.")
        
        species_to_idx = self.train_dataset.species_to_idx
        return {
            species_to_idx[species]: data.n_vars 
            for species, data in self.species_data.items()
        }


class CrossSpeciesInferenceDataset(IterableDataset):
    """Dataset for inference that sequentially processes all cells without sampling."""
    
    def __init__(
        self,
        species_data: Dict[str, ad.AnnData],
        batch_size: int = 128,
    ):
        """
        Initialize dataset.
        
        Args:
            species_data: Dictionary mapping species names to preprocessed AnnData objects
            batch_size: Batch size for processing
        """
        super().__init__()
        self.species_data = species_data
        self.batch_size = batch_size
        self.species_names = list(species_data.keys())
        self.species_to_idx = {name: idx for idx, name in enumerate(self.species_names)}


    def _create_sparse_batch(self, species: str, indices: np.ndarray):
        """Create a SparseExpressionData batch from indices."""
        species_adata = self.species_data[species]
        
        # Get batch data
        batch_data = species_adata[indices]
        batch_idx, gene_idx = batch_data.X.nonzero()
        values = torch.from_numpy(batch_data.X.data.astype(np.float32))
        gene_idx = torch.from_numpy(gene_idx.astype(np.int32))
        batch_idx = torch.from_numpy(batch_idx.astype(np.int32))
        
        # Create species_idx tensor
        species_idx = torch.full(
            (len(indices),), 
            self.species_to_idx[species], 
            dtype=torch.int32
        )
        
        return SparseExpressionData(
            values=values,
            batch_idx=batch_idx,
            gene_idx=gene_idx,
            species_idx=species_idx,
            batch_size=len(indices),
            n_genes=species_adata.n_vars,
            n_species=len(self.species_names),
        )

    def __iter__(self):
        """Iterate through all cells in each species sequentially."""
        for species in self.species_names:
            n_cells = self.species_data[species].n_obs
            
            # Process species in batches
            for start_idx in range(0, n_cells, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_cells)
                indices = np.arange(start_idx, end_idx)
                yield self._create_sparse_batch(species, indices)

    def __len__(self):
        """Return total number of batches across all species."""
        return sum(
            int(np.ceil(data.n_obs / self.batch_size))
            for data in self.species_data.values()
        )


class MultiSpeciesDataset(IterableDataset):
    def __init__(
        self,
        species_data: Dict[str, ad.AnnData],
        cluster_key: str,  # where cluster assignments are stored
        batch_size: int = 128,
        n_species_per_batch: int = 2,
        seed: int = 0,
        neighbors_per_cell: int = 100,
        max_num_steps: int | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_species_per_batch = n_species_per_batch
        self.species_data = species_data
        self.species_names = list(species_data.keys())
        self.species_to_idx = {name: idx for idx, name in enumerate(self.species_names)}
        self.rng = np.random.RandomState(seed)
        
        # Get cluster info for each species
        self.species_clusters = {
            species: set(adata.obs[cluster_key])
            for species, adata in species_data.items()
        }
        
        # Index cells by cluster for efficient sampling
        self.cluster_indices = {
            species: {
                cluster: np.where(adata.obs[cluster_key] == cluster)[0]
                for cluster in self.species_clusters[species]
            }
            for species, adata in species_data.items()
        }

        # Calculate max steps if not provided
        if max_num_steps is None:
            # Average number of cells across species
            avg_cells = np.mean([
                adata.n_obs 
                for adata in species_data.values()
            ])
            
            # Steps = (avg_cells * neighbors_per_cell) / batch_size
            self.max_num_steps = int((avg_cells * neighbors_per_cell) / batch_size)
        else:
            self.max_num_steps = max_num_steps
            
        print(f"Max steps per epoch: {self.max_num_steps}")


    def _create_sparse_batch(self, indices):
        """Create a SparseExpressionData batch from indices."""
        species = self.current_species
        species_adata = self.species_data[species]
        
        # Get batch data
        batch_data = species_adata[indices]
        batch_idx, gene_idx = batch_data.X.nonzero()
        values = torch.from_numpy(batch_data.X.data.astype(np.float32))
        gene_idx = torch.from_numpy(gene_idx.astype(np.int32))
        batch_idx = torch.from_numpy(batch_idx.astype(np.int32))
        

        species_idx = torch.full(
            (len(indices),), 
            self.species_to_idx[species], 
            dtype=torch.int32
        )
        
        return SparseExpressionData(
            values=values,
            batch_idx=batch_idx,
            gene_idx=gene_idx,
            species_idx=species_idx,
            batch_size=len(indices),
            n_genes=species_adata.n_vars,
            n_species=len(self.species_names),
        )

    def __len__(self):
        return self.max_num_steps
    
    def __iter__(self):
        while True:  # Infinite iterator since we're always resampling
            # Sample n_species_per_batch different species
            batch_species = self.rng.choice(
                self.species_names,
                size=self.n_species_per_batch,
                replace=False
            )
            
            # Create batch for each species
            batch = {}
            for species in batch_species:
                # Use diverse sampling instead of pure random sampling
                indices = self._sample_diverse_batch(species)
                self.current_species = species  # needed for _create_sparse_batch
                batch[self.species_to_idx[species]] = self._create_sparse_batch(indices)
            
            yield batch

    def _sample_diverse_batch(self, species):
        """Sample cells ensuring representation from different clusters"""
        n_clusters = len(self.species_clusters[species])
        
        if self.batch_size >= n_clusters:
            # Original logic for when batch_size >= n_clusters
            cells_per_cluster = self.batch_size // n_clusters
            indices = []
            for cluster in self.species_clusters[species]:
                cluster_cells = self.rng.choice(
                    self.cluster_indices[species][cluster],
                    size=cells_per_cluster,
                    replace=True
                )
                indices.extend(cluster_cells)
        else:
            # When batch_size < n_clusters:
            # 1. Randomly select batch_size clusters
            # 2. Take one cell from each selected cluster
            selected_clusters = self.rng.choice(
                list(self.species_clusters[species]),
                size=self.batch_size,
                replace=False
            )
            
            indices = []
            for cluster in selected_clusters:
                cell = self.rng.choice(
                    self.cluster_indices[species][cluster],
                    size=1,
                    replace=False
                )
                indices.extend(cell)
        
        # Fill any remaining slots
        remaining = self.batch_size - len(indices)
        if remaining > 0:
            extra_cells = self.rng.choice(
                len(self.species_data[species]),
                size=remaining,
                replace=True
            )
            indices.extend(extra_cells)
        
        return np.array(indices)
