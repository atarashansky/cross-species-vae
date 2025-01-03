from typing import Dict, Optional
from dataclasses import dataclass

import torch

@dataclass
class BatchData:
    data: Dict[int, torch.Tensor]
    labels: Optional[Dict[int, torch.Tensor]] = None
