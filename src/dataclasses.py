from typing import Dict
from dataclasses import dataclass

import torch

@dataclass
class BatchData:
    data: Dict[int, torch.Tensor]
