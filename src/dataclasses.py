from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np

@dataclass
class BatchData:
    data: Optional[Dict[int, torch.Tensor]] = None
    triplets: Optional[Dict[Tuple[int, int], Dict[str, torch.Tensor]]] = None
    labels: Optional[Dict[int, np.ndarray]] = None
