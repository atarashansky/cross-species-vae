from typing import Dict, Tuple
from dataclasses import dataclass

import torch

@dataclass
class BatchData:
    """Sparse representation of gene expression data."""

    data: torch.Tensor
    species_idx: int
