from typing import Dict
from dataclasses import dataclass

import torch

@dataclass
class BatchData:
    """Sparse representation of gene expression data."""
    data: Dict[int, torch.Tensor]
