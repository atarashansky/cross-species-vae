from typing import Dict, List, Optional, Union
import numpy as np
import anndata as ad
import pytorch_lightning as pl
import scipy as sp
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.dataclasses import BatchData


class CrossSpeciesDataset(IterableDataset):
    """Dataset for cross-species gene expression data."""

    def __init__(
        self,
        species_data: Dict[str, ad.AnnData],
        species_indices: Dict[str, np.ndarray],
        batch_size: int = 128,
        yield_pairwise: bool = False,
        subsample_size: Optional[int] = None,
        subsample_by: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize dataset.

        Args:
            species_data: Dictionary mapping species names to preprocessed AnnData objects
            species_indices: Dictionary mapping species names to array of valid indices
            batch_size: Number of cells per batch (per species)
            yield_pairwise: Whether to yield pairwise combinations of species
            subsample_size: If provided, subsample each species to this number of cells
            subsample_by: If provided, column name in obs metadata to use for stratified subsampling
        """
        super().__init__()
        self.batch_size = batch_size
        
        # Store species data directly
        self.species_data = species_data
        self.species_names = list(species_data.keys())
        
        # Create species indices
        self.species_to_idx = {name: idx for idx, name in enumerate(self.species_names)}
        
        # Create indices for each species
        self.species_indices = species_indices
        
        self.worker_id = None
        self.num_workers = None
        self.yield_pairwise = yield_pairwise
        self.subsample_size = subsample_size
        self.subsample_by = subsample_by
                
        # Initialize sampling state
        self._init_sampling_state()
        


    def _subsample_indices(self, species: str, indices: np.ndarray) -> np.ndarray:
        """Subsample indices, optionally maintaining proportions of a metadata column."""
        if self.subsample_size is None:
            return indices
        
        if self.subsample_by is None:
            # Simple random subsampling
            return self.rng.choice(indices, size=self.subsample_size, replace=False)
        
        # Stratified subsampling
        adata = self.species_data[species]
        labels = adata.obs[self.subsample_by[species]].iloc[indices]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Calculate target size for each label
        props = counts / len(indices)
        target_sizes = np.round(props * self.subsample_size).astype(int)
        
        # Adjust if rounding caused total to differ from target
        diff = self.subsample_size - target_sizes.sum()
        if diff != 0:
            # Add/subtract from largest group(s)
            idx_sorted = np.argsort(target_sizes)[::-1]
            for i in range(abs(diff)):
                target_sizes[idx_sorted[i]] += np.sign(diff)
        
        # Sample from each group
        sampled_indices = []
        remaining_target = self.subsample_size
        available_indices = indices.copy()
        
        for label, target_size in zip(unique_labels, target_sizes):
            label_indices = indices[labels == label]
            
            if target_size >= len(label_indices):
                # Take all available cells from this group
                sampled_indices.append(label_indices)
                remaining_target -= len(label_indices)
                # Remove these indices from available pool
                available_indices = np.setdiff1d(available_indices, label_indices)
            else:
                # Sample without replacement
                sampled = self.rng.choice(label_indices, size=target_size, replace=False)
                sampled_indices.append(sampled)
                remaining_target -= target_size
                # Remove these indices from available pool
                available_indices = np.setdiff1d(available_indices, sampled)
        
        # If we still need more cells, sample from all indices
        if remaining_target > 0:
            # If no available indices left, resample from all indices
            if len(available_indices) < remaining_target:
                extra_samples = available_indices
            else:
                extra_samples = self.rng.choice(available_indices, size=remaining_target, replace=False)
            sampled_indices.append(extra_samples)
        
        return np.concatenate(sampled_indices)

    def _init_sampling_state(self):
        """Initialize sampling state for each species."""
        # Generate new random seed for each epoch
        new_seed = np.random.randint(1e9)
        self.rng = np.random.RandomState(new_seed)
        
        # Subsample indices for each species
        available_indices = {
            species: self._subsample_indices(species, indices.copy())
            for species, indices in self.species_indices.items()
        }
        
        # Subset AnnData objects for the epoch and create new indices
        self.epoch_data = self.species_data
        self.epoch_indices = available_indices
        self.n_cells_per_species = {k: len(v) for k, v in available_indices.items()}
        
        self.seen_largest_dataset = set()
        
        # Find species with max cells for epoch tracking
        self.largest_species = max(
            ((species, len(data)) for species, data in self.epoch_data.items()),
            key=lambda x: x[1]
        )[0]

    def _get_batch_indices(self):
        """Get batch of indices for current species."""
        species = self.current_species
        available = self.epoch_indices[species]
        
        if len(available) >= self.batch_size:
            selected_idx = self.rng.choice(
                available, 
                self.batch_size, 
                replace=False
            )
            self.epoch_indices[species] = np.setdiff1d(
                available, 
                selected_idx
            )
            
            # Update seen indices for largest dataset
            if species == self.largest_species:
                self.seen_largest_dataset.update(selected_idx)
        else:
            selected_idx = available
            num_missing = self.batch_size - len(selected_idx)
            
            if species == self.largest_species:
                self.seen_largest_dataset.update(selected_idx)
            
            # Use indices within range of epoch_data
            max_idx = self.n_cells_per_species[species] - 1
            selected_idx = np.concatenate([
                selected_idx,
                self.rng.choice(
                    np.arange(max_idx + 1), 
                    num_missing, 
                    replace=True
                )
            ])
        
        return np.array(selected_idx)

    def _epoch_finished(self):
        """Check if we've seen all cells from the largest dataset."""
        return len(self.seen_largest_dataset) >= self.n_cells_per_species[self.largest_species]

    def _create_batch(self, indices):
        """Create batch from pre-subsetted epoch data."""
        species = self.current_species
        return torch.from_numpy(self.epoch_data[species].X[indices].toarray())

    def _initialize_worker_info(self):
        """Initialize worker information for distributed training."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

    
    def __len__(self):
        """Return the number of batches per epoch."""
        cells_in_largest = self.n_cells_per_species[self.largest_species]
        batches_for_largest = int(np.ceil(cells_in_largest / self.batch_size))
        len_multiplier = len(self.species_names) * (len(self.species_names) - 1) // 2 if self.yield_pairwise else 1
        return batches_for_largest * len_multiplier
    
    def __iter__(self):
        """Iterate over species in round-robin fashion."""
        self._initialize_worker_info()
        self._init_sampling_state()

        while not self._epoch_finished():
            data = {}
            for species in self.species_names:
                self.current_species = species
                indices = self._get_batch_indices()
                species_idx = self.species_to_idx[species]
                data[species_idx] = self._create_batch(indices)
            
            if self.yield_pairwise:
                for i in range(len(data)):
                    for j in range(i + 1, len(data)):
                        yield BatchData(data={
                            i: data[i],
                            j: data[j]
                        })
            else:
                yield BatchData(data=data)

class CrossSpeciesDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for cross-species data."""

    def __init__(
        self,
        species_data: Dict[str, Union[str, List[str], ad.AnnData, List[ad.AnnData]]],
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        yield_pairwise: bool = False,
        subsample_size: Optional[int] = None,
        subsample_by: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize data module.

        Args:
            species_data: Dictionary mapping species to data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            yield_pairwise: Whether to yield pairwise combinations of species
            subsample_size: If provided, subsample each species to this number of cells
            subsample_by: If provided, dictionary mapping species to column names in obs metadata
                         for stratified subsampling
        """
        super().__init__()
        self.species_data = species_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.yield_pairwise = yield_pairwise
        self.subsample_size = subsample_size
        self.subsample_by = subsample_by

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
        train_data_idx = {}
        val_data_idx = {}
        test_data_idx = {}
        
        for species, data in self.species_data.items():
            # Load data for this species
            species_adata = self._load_species_data(data)
            
            # Split the data
            train_frac = 1.0 - self.val_split - self.test_split
            train_idx, val_idx, test_idx = self._split_species_data(
                species_adata, 
                train_frac=train_frac,
                val_frac=self.val_split,
                test_frac=self.test_split
            )
            
            # Create split datasets
            train_data_idx[species] = train_idx
            val_data_idx[species] = val_idx
            test_data_idx[species] = test_idx
        
        
        if stage == "fit" or stage is None:
            self.train_dataset = CrossSpeciesDataset(
                species_data=self.species_data,
                species_indices=train_data_idx,
                batch_size=self.batch_size,
                yield_pairwise=self.yield_pairwise,
                subsample_size=self.subsample_size,
                subsample_by=self.subsample_by,
            )
            
            self.val_dataset = CrossSpeciesDataset(
                species_data=self.species_data,
                species_indices=val_data_idx,
                batch_size=self.batch_size,
                yield_pairwise=self.yield_pairwise,
                subsample_size=self.subsample_size,
                subsample_by=self.subsample_by,
            )

        if stage == "test" or stage is None:
            self.test_dataset = CrossSpeciesDataset(
                species_data=self.species_data,
                species_indices=test_data_idx,
                batch_size=self.batch_size,
                yield_pairwise=self.yield_pairwise,
                subsample_size=self.subsample_size,
                subsample_by=self.subsample_by,
            )
        

    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
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


    def _create_batch(self, species: str, indices: np.ndarray):
        """Create a BatchData batch from indices."""
        species_adata = self.species_data[species]     
        return torch.from_numpy(species_adata.X[indices].toarray())

    def __iter__(self):
        """Iterate through all cells in each species sequentially."""
        for species in self.species_names:
            n_cells = self.species_data[species].n_obs
            
            # Process species in batches
            for start_idx in range(0, n_cells, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_cells)
                indices = np.arange(start_idx, end_idx)
                data = self._create_batch(species, indices)
                yield BatchData(data={self.species_to_idx[species]: data})

    def __len__(self):
        """Return total number of batches across all species."""
        return sum(
            int(np.ceil(data.n_obs / self.batch_size))
            for data in self.species_data.values()
        )

