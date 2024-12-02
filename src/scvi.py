from typing import Dict, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils import negative_binomial_loss
from src.modules import GeneImportanceModule
from src.vae import CrossSpeciesVAE
from src.dataclasses import BatchData
from src.data import CrossSpeciesInferenceDataset

class SCVINetwork(pl.LightningModule):
    def __init__(
        self,
        cross_species_vae: CrossSpeciesVAE,
        n_hidden: int = 128,
        n_latent: int = 32,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        warmup_epochs: float = 0.1,
        init_beta: float = 1e-3,  # Start with very small KL weight
        final_beta: float = 1.0,  # Gradually increase to this value
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
        
        # Gene importance module
        self.gene_importance = GeneImportanceModule(self.input_dim, n_hidden)
        
        # Encoder with deep injection
        self.encoder = DeepInjectionEncoder(
            input_dim=self.input_dim,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_batch=n_species,  # use number of species as batch size
            dropout_rate=dropout_rate,
        )
        
        # Decoder with deep injection
        self.decoder = DeepInjectionDecoder(
            n_latent=n_latent,
            n_hidden=n_hidden,
            output_dim=self.input_dim,
            n_layers=n_layers,
            n_batch=n_species,
            dropout_rate=dropout_rate,
        )
        
        # Dispersion parameters for negative binomial
        self.theta = nn.Parameter(torch.ones(self.input_dim) * 2.0)
        
        # Store parameters
        self.learning_rate = learning_rate
        
        # Validation step outputs
        self.validation_step_outputs = []
        
    def preprocess_batch(self, batch: BatchData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform batch through CrossSpeciesVAE and concatenate outputs."""
        with torch.no_grad():
            # For each source species
            all_reconstructions = []
            batch_labels = []
            
            for source_species_id, source_data in batch.data.items():
                # Encode source data
                source_encoder = self.cross_species_vae.encoders[str(source_species_id)]
                source_encoded = source_encoder(source_data)
                
                # Get number of cells for batch labels
                n_cells = source_data.shape[0]
                
                # Decode into all species spaces
                cell_reconstructions = []
                for target_species_id in sorted(self.cross_species_vae.species_vocab_sizes.keys()):
                    target_decoder = self.cross_species_vae.decoders[str(target_species_id)]
                    reconstruction = target_decoder(source_encoded['z'])
                    cell_reconstructions.append(reconstruction['mean'])
                
                # Concatenate all target species reconstructions
                concatenated = torch.cat(cell_reconstructions, dim=1)
                all_reconstructions.append(concatenated)
                
                # Create batch labels (species IDs)
                batch_labels.append(torch.full((n_cells,), source_species_id, device=concatenated.device))
            
            # Combine all cells
            x = torch.cat(all_reconstructions, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0)
            
            return x, batch_labels
    
    def encode(self, x: torch.Tensor, batch_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x + self.gene_importance(x)
        
        # Encode
        mu, logvar = self.encoder(x, batch_idx)
        
        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std if self.training else mu
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
    
    def decode(self, z: torch.Tensor, batch_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Decode
        decoded = self.decoder(z, batch_idx)
        
        # Apply activation for counts
        mean = F.softplus(decoded)
        
        return {
            'mean': mean,
            'theta': torch.exp(self.theta)
        }
    
    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Full forward pass
        encoded = self.encode(x, batch_idx)
        decoded = self.decode(encoded['z'], batch_idx)
        
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
        x, batch_labels = self.preprocess_batch(batch)
        
        # Forward pass through SCVI
        outputs = self(x, batch_labels)
        
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
        x, batch_labels = self.preprocess_batch(batch)
        
        # Forward pass through SCVI
        outputs = self(x, batch_labels)
        
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
            x, batch_labels = self.preprocess_batch(batch)
            
            # Get latent representation from encoder
            encoded = self.encode(x, batch_labels)
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

class DeepInjectionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_hidden: int,
        n_latent: int,
        n_layers: int,
        n_batch: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Batch embedding shared across layers
        self.batch_embedding = nn.Embedding(n_batch, n_hidden)
        
        # Create layers with injection points
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        
        for i in range(n_layers):
            layer = nn.Sequential(
                nn.Linear(curr_dim + n_hidden, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer)
            curr_dim = n_hidden
        
        # Output layers
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)
        
    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get batch embedding
        batch_emb = self.batch_embedding(batch_idx)
        
        # Inject at each layer
        h = x
        for layer in self.layers:
            h = torch.cat([h, batch_emb], dim=1)
            h = layer(h)
        
        return self.mu(h), self.logvar(h)


class DeepInjectionDecoder(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_hidden: int,
        output_dim: int,
        n_layers: int,
        n_batch: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Separate batch embedding for decoder
        self.batch_embedding = nn.Embedding(n_batch, n_hidden)
        
        # Create layers with injection points
        self.layers = nn.ModuleList()
        curr_dim = n_latent
        
        for i in range(n_layers):
            layer = nn.Sequential(
                nn.Linear(curr_dim + n_hidden, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer)
            curr_dim = n_hidden
        
        # Output layer
        self.final = nn.Linear(n_hidden, output_dim)
        
    def forward(self, z: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        # Get batch embedding
        batch_emb = self.batch_embedding(batch_idx)
        
        # Inject at each layer
        h = z
        for layer in self.layers:
            h = torch.cat([h, batch_emb], dim=1)
            h = layer(h)
        
        return self.final(h)