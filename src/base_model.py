import pytorch_lightning as pl
import torch
import math


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        base_learning_rate: float = 1e-3,
        base_batch_size: int = 256,
        batch_size: int = 256,
        min_learning_rate: float = 1e-5,
        warmup_data: float = 0.1,
        init_beta: float = 1e-3,
        final_beta: float = 1,                 
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",        
    ):
        super().__init__()

        if init_beta > final_beta:
            init_beta = final_beta
            
        # Scale learning rate based on batch size
        self.learning_rate = base_learning_rate
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.current_beta = init_beta  # Track current beta value
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size

        
        # Training parameters
        self.min_learning_rate = min_learning_rate
        self.warmup_data = warmup_data

        self.warmup_steps = None  # Will be set in on_train_start
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.validation_step_outputs = []


    def configure_optimizers(self):
        # Calculate dataset size from total steps
        total_steps = self.trainer.estimated_stepping_batches
        num_epochs = self.trainer.max_epochs
        steps_per_epoch = total_steps // num_epochs
        dataset_size = steps_per_epoch * self.hparams.batch_size
        # Determine scaling method based on batch/dataset ratio
        batch_ratio = self.hparams.batch_size / dataset_size
        if batch_ratio > 0.1:  # Using 10% as threshold
            scaling_factor = self.hparams.batch_size / self.hparams.base_batch_size  # Linear scaling
        else:
            scaling_factor = math.sqrt(self.hparams.batch_size / self.hparams.base_batch_size)  # Sqrt scaling
            
        # Apply scaling to learning rate
        scaled_lr = self.learning_rate * scaling_factor

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=scaled_lr,
            weight_decay=0.01,
        )
        
        def lr_lambda(current_step: int):
            if self.warmup_steps is None:
                return 1.0
            
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                )
                return max(
                    self.min_learning_rate / self.learning_rate,
                    0.5 * (1.0 + math.cos(math.pi * progress * 0.85))
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self):
        if self.warmup_steps is None:
            steps_per_epoch = len(self.trainer.train_dataloader)
            total_samples = steps_per_epoch * self.batch_size
            target_warmup_samples = total_samples * self.warmup_data
            self.warmup_steps = int(target_warmup_samples / self.batch_size)
            self.warmup_steps = max(100, self.warmup_steps)

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """Configure gradient clipping with improved settings."""
        gradient_clip_val = gradient_clip_val or self.gradient_clip_val
        gradient_clip_algorithm = gradient_clip_algorithm or self.gradient_clip_algorithm

        # TODO: try this out
        # # Add gradient noise scaled by batch size
        # if hasattr(self, 'hparams') and hasattr(self.hparams, 'batch_size'):
        #     noise_scale = math.sqrt(self.hparams.base_batch_size / self.hparams.batch_size)
        #     for param in self.parameters():
        #         if param.grad is not None:
        #             param.grad += torch.randn_like(param.grad) * noise_scale * 1e-5

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return
        
        metrics = {
            key: torch.stack([x[key] if isinstance(x[key], torch.Tensor) else torch.tensor(x[key], device=self.device) for x in self.validation_step_outputs]).mean()
            for key in self.validation_step_outputs[0].keys()
        }

        for key, value in metrics.items():
            self.log(key, value, sync_dist=True)

        self.validation_step_outputs.clear()

    def get_current_beta(self) -> float:
        """Calculate current beta value based on training progress"""
        if self.trainer is None:
            return self.init_beta
            
        # Adjust for batch size effect on number of steps
        current_step = self.trainer.global_step
        total_steps = self.trainer.estimated_stepping_batches
        
        # Linear scaling with batch size
        batch_size_factor = self.hparams.batch_size / self.hparams.base_batch_size
        progress = (current_step * batch_size_factor) / total_steps
        
        # Linear warmup from init_beta to final_beta
        beta = self.init_beta + (self.final_beta - self.init_beta) * progress
        return beta
