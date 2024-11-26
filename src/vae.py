from typing import Dict, Any, Union, List, Optional, Tuple
import anndata as ad
import pytorch_lightning as pl
import torch
from torch import cuda
import math
import torch.nn as nn
import torch.nn.functional as F

from src.dataclasses import SparseExpressionData
from src.data import CrossSpeciesDataModule

class GeneImportanceModule(nn.Module):
    def __init__(self, n_genes: int, n_hidden: int = 256):
        super().__init__()
        # Global gene importance weights
        self.global_weights = nn.Parameter(torch.ones(n_genes))
        
        # Context network - using smaller hidden dimensions
        self.context_net = nn.Sequential(
            nn.Linear(n_genes, n_hidden),  # Reduce to smaller dimension first
            nn.ReLU(),
            nn.Linear(n_hidden, n_genes),  # Project back to gene space
            nn.Softplus()
        )
        
        # Initialize global weights near 1
        nn.init.normal_(self.global_weights, mean=1.0, std=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get context-dependent weights
        context_weights = self.context_net(x)  # [batch_size, n_genes]
        
        # Combine global and context weights
        importance = self.global_weights * context_weights
        
        return x * importance


class Encoder(nn.Module):
    """Multi-scale encoder with species-specific and global latent spaces."""

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_latent: int,
        hidden_dims: list,
        n_context_hidden: int = 256,
        species_embedding_dim: int = 32,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Gene importance module
        self.gene_importance = GeneImportanceModule(n_genes, n_context_hidden)
        
        # Species embedding
        self.species_embedding = nn.Embedding(n_species, species_embedding_dim)
        
        # Single input layer that handles both with/without species embedding
        self.input_layer = nn.Linear(n_genes + species_embedding_dim, hidden_dims[0])
        
        # Rest of the encoder
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        self.encoder_net = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_dims[-1], n_latent)
        self.logvar = nn.Linear(hidden_dims[-1], n_latent)

        # TODO: Try different initializations 
        # e.g. 
        # nn.init.xavier_normal_(self.mu.weight, gain=1.0)
        # nn.init.zeros_(self.mu.bias)
        # nn.init.xavier_normal_(self.logvar.weight, gain=0.1)
        # nn.init.constant_(self.logvar.bias, -2.0)
        
    def forward(self, batch: SparseExpressionData, use_species: bool = True) -> torch.Tensor:
        # Create dense gene expression matrix
        x = torch.zeros(
            batch.batch_size,
            batch.n_genes,
            device=batch.values.device,
            dtype=batch.values.dtype
        )
        x[batch.batch_idx, batch.gene_idx] = batch.values
        
        # Apply gene importance
        x = self.gene_importance(x)
        
        if use_species:
            # Use full input layer with species embedding
            species_emb = self.species_embedding(batch.species_idx)
            x = torch.cat([x, species_emb], dim=1)
            h = self.input_layer(x)
        else:
            # Use only the gene expression part of input layer weights
            h = F.linear(
                x, 
                self.input_layer.weight[:, :batch.n_genes],  # Only use gene weights
                self.input_layer.bias
            )
        
        # Continue with rest of encoder
        h = self.encoder_net(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
            
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        hidden_dims: list,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Decoder layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(n_latent, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], n_genes))
        
        self.decoder_net = nn.Sequential(*layers)
        
        # Separate activation for non-negative output
        self.activation = nn.Softplus()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Decode
        h = self.decoder_net(z)
        
        # Apply activation for non-negative output
        return self.activation(h)


class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    def __init__(
        self,
        species_vocab_sizes: Dict[int, int],
        homology_edges: torch.Tensor,
        homology_scores: torch.Tensor,
        n_latent: int = 32,
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        warmup_epochs: float = 0.1,
        init_beta: float = 1.0,  # Initial KL weight
        min_beta: float = 0.0,   # Minimum beta value
        max_beta: float = 10.0,  # Maximum beta value
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Make beta a learnable parameter
        self.log_beta = nn.Parameter(torch.log(torch.tensor(init_beta)))
        self.min_beta = min_beta
        self.max_beta = max_beta
        
        # Store parameters
        self.species_vocab_sizes = species_vocab_sizes
        self.n_species = len(species_vocab_sizes)
        self.n_latent = n_latent
        
        # Create species-specific encoders/decoders
        self.encoders = nn.ModuleDict({
            str(species_id): Encoder(
                n_species=self.n_species,
                n_genes=vocab_size,
                n_latent=n_latent,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
            )
            for species_id, vocab_size in species_vocab_sizes.items()
        })
        
        self.decoders = nn.ModuleDict({
            str(species_id): Decoder(
                n_genes=vocab_size,
                n_latent=n_latent,
                hidden_dims=hidden_dims[::-1],
                dropout_rate=dropout_rate,
            )
            for species_id, vocab_size in species_vocab_sizes.items()
        })
        
        # Register homology information
        self.register_buffer("homology_edges", homology_edges)
        self.homology_scores = nn.Parameter(homology_scores.float())
        
        # Training parameters
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_epochs = warmup_epochs

        self.warmup_steps = None  # Will be set in on_train_start
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.validation_step_outputs = []

    
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
    
    @property
    def beta(self):
        """Get current beta value with constraints."""
        return torch.clamp(torch.exp(self.log_beta), self.min_beta, self.max_beta)

    def on_train_start(self):
        """Calculate warmup steps when training starts."""
        if self.warmup_steps is None:
            steps_per_epoch = len(self.trainer.train_dataloader)
            self.warmup_steps = int(steps_per_epoch * self.warmup_epochs)
            print(f"Warmup steps calculated: {self.warmup_steps}")

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

    def validation_step(self, batch: SparseExpressionData, batch_idx: int):
        """Validation step."""
        # Forward pass
        predictions, mu, logvar = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, predictions, mu, logvar)
        
        # Store for epoch end processing
        self.validation_step_outputs.append({
            "val_loss": loss_dict["loss"],
            "val_recon_loss": loss_dict["recon_loss"],
            "val_kl_loss": loss_dict["kl_loss"],
            "val_homology_loss": loss_dict["homology_loss"],
        })
        
        return loss_dict["loss"]

    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return

        metrics = {
            key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
            for key in self.validation_step_outputs[0].keys()
        }

        for key, value in metrics.items():
            self.log(key, value, sync_dist=True)

        self.validation_step_outputs.clear()

    def on_train_epoch_start(self):
        """Reset memory tracking at the start of each epoch."""
        self.prev_gpu_memory = 0
        if cuda.is_available():
            cuda.empty_cache()

    def compute_loss(
        self, 
        batch: SparseExpressionData,
        predictions: Dict[int, torch.Tensor],
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        
        # Get species ID for this batch
        species_id = batch.species_idx[0].item()
        
        # Create dense target tensor
        target = torch.zeros(
            batch.batch_size,
            batch.n_genes,
            device=batch.values.device,
            dtype=batch.values.dtype
        )
        target[batch.batch_idx, batch.gene_idx] = batch.values
        
        # 1. Reconstruction loss (Poisson NLL)
        recon_loss = F.poisson_nll_loss(
            predictions[species_id].clamp(min=1e-6),
            target,  # Use dense tensor
            log_input=False,
            full=True,
            reduction="mean"
        )
        
        # 2. KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # 3. Homology consistency loss
        homology_loss = self.compute_homology_loss(predictions, batch)
        
        # 4. Total loss with learned beta
        total_loss = recon_loss + self.beta * kl_loss + homology_loss
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "homology_loss": homology_loss,
        }

    def training_step(self, batch: SparseExpressionData, batch_idx: int):
        """Training step with automatic optimization."""
        # Forward pass
        predictions, mu, logvar = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, predictions, mu, logvar)
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"], sync_dist=True)
        self.log("train_recon_loss", loss_dict["recon_loss"], sync_dist=True)
        self.log("train_kl_loss", loss_dict["kl_loss"], sync_dist=True)
        self.log("train_homology_loss", loss_dict["homology_loss"], sync_dist=True)
        self.log("beta", self.beta, sync_dist=True)
        
        return loss_dict["loss"]

    def compute_homology_loss(self, predictions: Dict[int, torch.Tensor], batch: SparseExpressionData) -> torch.Tensor:
        # Compute consistency loss between homologous genes
        homology_loss = 0.0
        src, dst = self.homology_edges.t()
        edge_weights = F.softmax(self.homology_scores, dim=0)
        
        for i, (src_gene, dst_gene) in enumerate(zip(src, dst)):
            src_species = src_gene // self.species_vocab_sizes[0]  # Assuming sorted vocab sizes
            dst_species = dst_gene // self.species_vocab_sizes[0]
            
            if src_species in predictions and dst_species in predictions:
                src_pred = predictions[src_species][:, src_gene]
                dst_pred = predictions[dst_species][:, dst_gene]
                
                # MSE between homologous gene predictions
                gene_loss = F.mse_loss(src_pred, dst_pred)
                homology_loss += edge_weights[i] * gene_loss
        
        return homology_loss

    
    def forward(self, batch: SparseExpressionData, use_species: bool = True) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        # Get species ID for this batch
        species_id = batch.species_idx[0].item()
        
        # Forward pass through encoder
        encoder = self.encoders[str(species_id)]
        z, mu, logvar = encoder(batch, use_species=use_species)
        
        # Get predictions for all species
        predictions = {}
        for decoder_species_id in self.decoders:
            predictions[int(decoder_species_id)] = self.decoders[decoder_species_id](z)
        
        return predictions, mu, logvar

    @torch.no_grad()
    def get_latent_embeddings(
        self,
        species_data: Dict[str, Union[str, List[str], ad.AnnData, List[ad.AnnData]]],
        use_species: bool = False,
        return_species: bool = True,
        batch_size: int = 512,
        num_workers: int = 4,
        device: Optional[torch.device] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get latent embeddings for cells in provided AnnData objects.
        
        Args:
            species_data: Dictionary mapping species names to AnnData objects
                (same format as CrossSpeciesDataModule input)
            use_species: Whether to use species information in encoding
            return_species: Whether to return species labels
            batch_size: Batch size for processing
            num_workers: Number of dataloader workers
            device: Device to use for computation. If None, uses model's device
            
        Returns:
            If return_species:
                latents: Tensor of shape [n_cells, n_latent]
                species: Tensor of shape [n_cells] containing species indices
            Else:
                latents: Tensor of shape [n_cells, n_latent]
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Create data module with no train/val/test split
        data_module = CrossSpeciesDataModule(
            species_data=species_data,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=0.0,
            test_split=0.0,
            seed=0,
        )
        
        # Get full dataset
        data_module.setup()
        dataset = data_module.train_dataset  # Contains all data since splits are 0
        
        self.eval()  # Set to evaluation mode
        
        all_latents = []
        all_species = []
        
        for batch in dataset:
            # Move batch to device
            batch = SparseExpressionData(
                values=batch.values.to(device),
                batch_idx=batch.batch_idx.to(device),
                gene_idx=batch.gene_idx.to(device),
                species_idx=batch.species_idx.to(device),
                batch_size=batch.batch_size,
                n_genes=batch.n_genes,
                n_species=batch.n_species,
            )
            
            # Get species-specific encoder
            species_id = batch.species_idx[0].item()
            
            # Get latent embeddings
            latents, _, _ = self.encoders[str(species_id)](batch, use_species=use_species)
            
            all_latents.append(latents.cpu())
            if return_species:
                # Get one species label per cell
                species_per_cell = torch.zeros(batch.batch_size, dtype=torch.long)
                species_per_cell[:] = species_id
                all_species.append(species_per_cell)
        
        # Concatenate all batches
        latents = torch.cat(all_latents, dim=0)
        
        if return_species:
            species = torch.cat(all_species, dim=0)
            return latents, species
        
        return latents



