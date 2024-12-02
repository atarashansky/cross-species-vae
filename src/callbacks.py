import torch
from pytorch_lightning.callbacks import Callback
import logging

class StageAwareEarlyStopping(Callback):
    """Early stopping callback that only monitors metrics during relevant stages."""
    
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = 'min',
        check_on_train_epoch_end: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.check_on_train_epoch_end = check_on_train_epoch_end
        self.verbose = verbose
        
        # Initialize tracking variables
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.stop_training = False
        
        # Set mode-specific variables
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode {mode} is not supported")
        
        self.monitor_op = torch.lt if mode == 'min' else torch.gt
        self.min_delta *= 1 if mode == 'min' else -1
        
    def _should_monitor_current_stage(self, trainer) -> bool:
        """Determine if we should monitor the metric in the current stage."""
        current_stage = trainer.lightning_module.get_stage()
        
        # Map metrics to their relevant stages
        metric_stage_mapping = {
            'val_recon': ['direct_recon', 'transform_recon'],
            'val_kl': ['direct_recon', 'transform_recon'],
            'val_homology': ['homology_loss'],
            'val_loss': ['direct_recon', 'transform_recon', 'homology_loss'],
        }
        
        relevant_stages = metric_stage_mapping.get(self.monitor, [])
        return current_stage in relevant_stages
    
    def _get_monitor_value(self, logs: dict) -> torch.Tensor | None:
        """Get the monitored value from logs."""
        monitor_val = logs.get(self.monitor)
        if monitor_val is None:
            return None
            
        if isinstance(monitor_val, torch.Tensor):
            monitor_val = monitor_val.cpu()
        return torch.tensor(monitor_val, dtype=torch.float32)
    
    def on_validation_end(self, trainer, pl_module):
        """Check early stopping criteria after validation."""
        if not self._should_monitor_current_stage(trainer):
            return
        logs = trainer.callback_metrics
        monitor_val = self._get_monitor_value(logs)
        
        if monitor_val is None:
            return
        
        if self.best_score is None:
            self.best_score = monitor_val
        elif self.monitor_op(monitor_val - self.min_delta, self.best_score):
            self.best_score = monitor_val
            self.wait_count = 0
            print(f'Metric {self.monitor} improved to {self.best_score}')
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                self.stop_training = True
                trainer.should_stop = True
                
                if self.verbose:
                    stage = trainer.lightning_module.get_stage()
                    print(f'Early stopping triggered in stage {stage} at epoch {self.stopped_epoch}')