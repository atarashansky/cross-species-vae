from typing import Dict, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils import negative_binomial_loss
from src.modules import Encoder, Decoder
from src.multi_vae import CrossSpeciesVAE
from src.dataclasses import BatchData
from src.data import CrossSpeciesInferenceDataset

class VAE(pl.LightningModule):
    def __init__(
        self,
        cross_species_vae: CrossSpeciesVAE,
        n_latent: int = 128,
        hidden_dims: list = [128],
        dropout_rate: float = 0.1,
        base_learning_rate: float = 1e-3,
        base_batch_size: int = 32,
        batch_size: int | None = None,
        min_learning_rate: float = 1e-5,
        warmup_epochs: float = 0.1,
        init_beta: float = 1e-3,
        final_beta: float = 0.1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['cross_species_vae'])
        
        # Training parameters
        self.min_learning_rate = min_learning_rate
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = None
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        
        # Store and freeze the VAE
        self.cross_species_vae = cross_species_vae
        self.cross_species_vae.eval()
        for param in self.cross_species_vae.parameters():
            param.requires_grad = False
            
        # Calculate input dimension (sum of all species gene spaces)
        self.input_dim = sum(vocab_size for vocab_size in cross_species_vae.species_vocab_sizes.values())
        n_species = len(cross_species_vae.species_vocab_sizes)
        if batch_size is not None:
            self.learning_rate = base_learning_rate * (batch_size / base_batch_size)
        else:
            self.learning_rate = base_learning_rate
        

        self.mu_layer = nn.Linear(hidden_dims[-1], n_latent)
        self.logvar_layer = nn.Linear(hidden_dims[-1], n_latent)
     
        self.encoder = Encoder(
            n_genes=self.input_dim,
            mu_layer=self.mu_layer,
            logvar_layer=self.logvar_layer,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )   

        self.decoder = Decoder(
            n_genes=self.input_dim,
            n_latent=n_latent,
            hidden_dims=hidden_dims[::-1],  # Reverse hidden dims for decoder
            dropout_rate=dropout_rate,
        )
        
        self.validation_step_outputs = []
        
    def preprocess_batch(self, batch: BatchData) -> torch.Tensor:
        """Transform batch through CrossSpeciesVAE and concatenate outputs."""
        with torch.no_grad():
            # For each source species
            all_reconstructions = []
            
            for source_species_id, source_data in batch.data.items():
                # Encode source data
                source_encoder = self.cross_species_vae.encoders[str(source_species_id)]
                source_encoded = source_encoder(source_data)
                                
                # Decode into all species spaces
                cell_reconstructions = []
                for target_species_id in sorted(self.cross_species_vae.species_vocab_sizes.keys()):
                    target_decoder = self.cross_species_vae.decoders[str(target_species_id)]
                    reconstruction = target_decoder(source_encoded['z'])
                    cell_reconstructions.append(reconstruction['mean'])
                
                # Concatenate all target species reconstructions
                concatenated = torch.cat(cell_reconstructions, dim=1)
                all_reconstructions.append(concatenated)
                        
            return torch.cat(all_reconstructions, dim=0)
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Full forward pass
        encoded = self.encode(x)
        decoded = self.decode(encoded['z'])
        
        return {
            **encoded,
            **decoded
        }
    
    def _compute_reconstruction_loss(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        return negative_binomial_loss(
            pred=mean,
            target=x,
            theta=theta
        )
    
    def _compute_kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    
    def training_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        # Preprocess batch through frozen VAE
        x = self.preprocess_batch(batch)
        
        # Forward pass through SCVI
        outputs = self(x)
        
        # Compute losses
        recon_loss = self._compute_reconstruction_loss(
            x=x,
            mean=outputs['mean'],
            theta=outputs['theta']
        )
        
        kl_loss = self._compute_kl_loss(
            mu=outputs['mu'],
            logvar=outputs['logvar']
        )
        
        # Get current beta for KL weight
        beta = self.get_current_beta()
        
        # Compute losses with current beta
        total_loss = recon_loss + beta * kl_loss
        
        # Log metrics
        self.log('train_loss', total_loss)
        self.log('train_recon', recon_loss)
        self.log('train_kl', kl_loss)
        
        return total_loss
    
    def validation_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        # Preprocess batch through frozen VAE
        x = self.preprocess_batch(batch)
        
        # Forward pass through SCVI
        outputs = self(x)
        
        # Compute losses
        recon_loss = self._compute_reconstruction_loss(
            x=x,
            mean=outputs['mean'],
            theta=outputs['theta']
        )
        
        kl_loss = self._compute_kl_loss(
            mu=outputs['mu'],
            logvar=outputs['logvar']
        )
        
        # Total loss
        total_loss = recon_loss + self.get_current_beta() * kl_loss
        
        # Store outputs for epoch end averaging
        self.validation_step_outputs.append({
            "val_loss": total_loss.detach(),
            "val_recon": recon_loss.detach(),
            "val_kl": kl_loss.detach(),
        })
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Compute average validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return
        
        # Calculate mean of all metrics
        metrics = {
            key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
            for key in self.validation_step_outputs[0].keys()
        }
        
        # Log averaged metrics
        for key, value in metrics.items():
            self.log(key, value, sync_dist=True)
        
        # Clear saved outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        def lr_lambda(current_step: int):
            if self.warmup_steps is None:
                return 1.0
            
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay with minimum learning rate
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                )
                return max(
                    self.min_learning_rate / self.learning_rate,
                    0.5 * (1.0 + math.cos(math.pi * progress)),
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

    @torch.no_grad()
    def get_latent_embeddings(
        self,
        species_data,
        batch_size: int = 512,
        device = "cuda",
    ) -> torch.Tensor:
        """
        Get latent embeddings for cells in provided AnnData objects.
        
        Args:
            species_data: Dictionary mapping species names to AnnData objects
            batch_size: Batch size for processing
            device: Device to use for computation. If None, uses model's device
            
        Returns:
            latents: Tensor of shape [n_cells, n_latent]
        """
        if device is None:
            device = next(self.parameters()).device
        elif device is not None:
            self.to(device)
            
        dataset = CrossSpeciesInferenceDataset(
            species_data=species_data,
            batch_size=batch_size,
        )
        
        # Set to evaluation mode
        self.eval()
        all_latents = []
        
        for batch in dataset:
            # Move batch to device and preprocess through VAE
            batch = BatchData(
                data={k: v.to(device) for k,v in batch.data.items()},
            )
            x = self.preprocess_batch(batch)
            
            # Get latent representation from encoder
            encoded = self.encode(x)
            all_latents.append(encoded['z'].cpu())
        
        # Concatenate all batches
        latents = torch.cat(all_latents, dim=0)
        return latents

    def on_train_start(self):
        """Calculate warmup steps when training starts."""
        if self.warmup_steps is None:
            steps_per_epoch = len(self.trainer.train_dataloader)
            self.warmup_steps = int(steps_per_epoch * self.warmup_epochs)

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """Configure gradient clipping with improved settings."""
        gradient_clip_val = gradient_clip_val or self.gradient_clip_val
        gradient_clip_algorithm = gradient_clip_algorithm or self.gradient_clip_algorithm

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def get_current_beta(self) -> float:
        """Calculate current beta value based on training progress"""
        if self.trainer is None:
            return self.init_beta
            
        # Get current epoch and total epochs
        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs
        
        # Linear warmup from init_beta to final_beta
        progress = current_epoch / max_epochs
        beta = self.init_beta + (self.final_beta - self.init_beta) * progress
        return beta
