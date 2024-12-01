from typing import Dict, Any, Union, List, Optional, Tuple
import anndata as ad
import pytorch_lightning as pl
import torch
from torch import cuda
import math
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from src.dataclasses import BatchData
from src.data import CrossSpeciesInferenceDataset


class FrozenEmbeddingLayer(nn.Module):
    def __init__(self, embedding_weights: torch.Tensor, dropout: float = 0.1):  # [n_genes, embedding_dim]
        super().__init__()
        
        # Convert weights to float32 for better numerical stability
        embedding_weights = embedding_weights.to(torch.float32)
        
        self.embedding = nn.Linear(embedding_weights.shape[0], embedding_weights.shape[1], bias=False)
        self.embedding.weight.data = embedding_weights.T  # Transpose for matmul
        self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        # Add layer norm after embedding
        self.norm = nn.LayerNorm(embedding_weights.shape[1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is same dtype as weights
        x = x.to(self.embedding.weight.dtype)
        x = self.dropout(x)
        x = self.embedding(x)
        # Normalize embeddings
        return self.norm(x)
    
class GeneImportanceModule(nn.Module):
    def __init__(self, n_genes: int, n_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.global_weights = nn.Parameter(torch.ones(n_genes))

        self.dropout = nn.Dropout(dropout)
        self.context_net = nn.Sequential(
            nn.LayerNorm(n_genes),
            nn.Linear(n_genes, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            self.dropout,
            nn.Linear(n_hidden, n_genes),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_weights = torch.sigmoid(self.global_weights)
        context_weights = self.context_net(x)
        importance = global_weights * context_weights
        return x * importance

class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        mu_layer: nn.Linear,
        logvar_layer: nn.Linear,
        hidden_dims: list,
        n_context_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.gene_importance = GeneImportanceModule(n_genes, n_context_hidden)        
        
        # Create encoder layers helper
        def make_encoder_layers(input_dim, hidden_dims):
            layers = []
            dims = [input_dim] + hidden_dims
            
            for i in range(len(dims) - 1):
                layers.extend([
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
            return nn.Sequential(*layers)
        
        # Shared pathway
        self.encoder = make_encoder_layers(n_genes, hidden_dims)
        self.mu = mu_layer
        self.logvar = logvar_layer
        
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-20, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std if self.training else mu
        
    def forward(self, x: BatchData) -> Dict[str, torch.Tensor]:
        dense_x = x.data

        x = dense_x + self.gene_importance(dense_x)

        # Main encoding
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
        }
   

class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        species_embedding: nn.Embedding,
        hidden_dims: list,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.log_theta = nn.Parameter(torch.ones(n_genes) * 2.3)

        # Technical scaling network
        self.scaling_net = nn.Sequential(
            nn.Linear(n_latent + species_embedding.embedding_dim, n_genes),
            nn.Sigmoid()
        )
        
        # Biological decoder network
        layers = []
        dims = [n_latent + species_embedding.embedding_dim] + hidden_dims + [n_genes]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]) if i < len(dims) - 2 else nn.Identity(),
                nn.ReLU() if i < len(dims) - 2 else nn.Softplus(),
                nn.Dropout(dropout_rate) if i < len(dims) - 2 else nn.Identity()
            ])
            
        self.species_embedding = species_embedding
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z: Dict[str, torch.Tensor], species_idx: int) -> Dict[str, torch.Tensor]:
        # Get biological factors
        batch_species_idx = torch.full((z['z'].shape[0],), species_idx, dtype=torch.long, device=z['z'].device)
        species_embedding = self.species_embedding(batch_species_idx)
        z_input = torch.cat([z['z'], species_embedding], dim=1)
        bio_factors = self.decoder_net(z_input)
        
        # Get technical scaling factors
        scaling_factors = self.scaling_net(z_input)

        # Combine all factors
        mean = bio_factors * scaling_factors
        theta = torch.exp(self.log_theta)  # Ensure theta is positive
        
        return {
            'mean': mean,
            'theta': theta
        }

class SpeciesDiscriminator(nn.Module):
    def __init__(self, n_latent: int, n_species: int, hidden_dims: list = [64], dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        dims = [n_latent] + hidden_dims + [n_species]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]) if i < len(dims) - 2 else nn.Identity(),
                nn.ReLU() if i < len(dims) - 2 else nn.Identity(),
                nn.Dropout(dropout_rate) if i < len(dims) - 2 else nn.Identity()
            ])
            
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    def __init__(
        self,
        species_vocab_sizes: Dict[int, int],
        homology_edges: Dict[int, Dict[int, torch.Tensor]] | None = None,
        homology_scores: Dict[int, Dict[int, torch.Tensor]] | None = None,
        n_latent: int = 32,
        hidden_dims: list = [128],
        dropout_rate: float = 0.1,
        base_learning_rate: float = 1e-3,
        base_batch_size: int = 32,
        batch_size: int | None = None,
        min_learning_rate: float = 1e-5,
        species_embedding_dim: int = 64,
        warmup_epochs: float = 0.1,
        init_beta: float = 1e-3,
        final_beta: float = 0.1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        recon_weight: float = 1.0,
        adversarial_weight: float = 0.1,  # Add this parameter
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['homology_edges'])
        
        # Scale learning rate based on batch size
        if batch_size is not None:
            self.learning_rate = base_learning_rate * (batch_size / base_batch_size)
        else:
            self.learning_rate = base_learning_rate
                    
        self.recon_weight = recon_weight

        self.init_beta = init_beta
        self.final_beta = final_beta
        self.current_beta = init_beta  # Track current beta value
        
        # Store parameters
        self.species_vocab_sizes = species_vocab_sizes
        self.n_species = len(species_vocab_sizes)
        self.n_latent = n_latent

        self.mu_layer = nn.Linear(hidden_dims[-1], n_latent)
        self.logvar_layer = nn.Linear(hidden_dims[-1], n_latent)
        
        self.decoder_species_embedding = nn.Embedding(self.n_species, species_embedding_dim)
        
        # Create species-specific encoders/decoders
        self.encoders = nn.ModuleDict({
            str(species_id): Encoder(
                n_genes=vocab_size,
                mu_layer=self.mu_layer,
                logvar_layer=self.logvar_layer,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            )
            for species_id, vocab_size in species_vocab_sizes.items()
        })
    
        self.decoders = nn.ModuleDict({
            str(species_id): Decoder(
                n_genes=vocab_size,
                n_latent=n_latent,
                species_embedding=self.decoder_species_embedding,
                hidden_dims=hidden_dims[::-1],  # Reverse hidden dims for decoder
                dropout_rate=dropout_rate,
            )
            for species_id, vocab_size in species_vocab_sizes.items()
        })

        # Register homology information with learnable scores
        if homology_edges is not None:
            self.homology_edges = homology_edges
            
            # Initialize learnable scores for each species pair
            self.homology_scores = nn.ParameterDict({
                str(src_id): nn.ParameterDict({
                    str(dst_id): nn.Parameter(
                        torch.ones(len(edges), dtype=torch.float32) if homology_scores is None else homology_scores[src_id][dst_id] / homology_scores[src_id][dst_id].max()
                    )
                    for dst_id, edges in species_edges.items()
                })
                for src_id, species_edges in homology_edges.items()
            })
        
        # Training parameters
        self.min_learning_rate = min_learning_rate
        self.warmup_epochs = warmup_epochs

        self.warmup_steps = None  # Will be set in on_train_start
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.validation_step_outputs = []
        
        self.adversarial_weight = adversarial_weight

        self.discriminator = SpeciesDiscriminator(
            n_latent=n_latent,
            n_species=self.n_species,
            dropout_rate=dropout_rate
        )
    
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
    

    def training_step(self, batch: BatchData, batch_idx: int):
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"].detach(), sync_dist=True)
        self.log("train_recon", loss_dict["recon"].detach(), sync_dist=True)
        self.log("train_kl", loss_dict["kl"].detach(), sync_dist=True)
        self.log("train_adversarial", loss_dict["adversarial"].detach(), sync_dist=True)
        
        return loss_dict["loss"]
        
    def validation_step(self, batch: BatchData, batch_idx: int):
        """Validation step."""
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        
        # Store only the scalar values, detached from computation graph
        self.validation_step_outputs.append({
            "val_loss": loss_dict["loss"].detach(),
            "val_recon": loss_dict["recon"].detach(),
            "val_kl": loss_dict["kl"].detach(),
            "val_adversarial": loss_dict["adversarial"].detach(),
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
   
    def _compute_reconstruction_loss(
        self,
        target_input: torch.Tensor,
        reconstruction: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        # Convert from log space to count space
        target_counts = torch.exp(target_input) - 1  # If log1p was used
        pred_counts = torch.exp(reconstruction['mean']) - 1
        
        # count_loss_poisson = (self.species_vocab_sizes[0] / vocab_size) * F.poisson_nll_loss(
        #     pred_counts.clamp(min=1e-6),
        #     target_counts,
        #     log_input=False,
        #     full=True,
        #     reduction="mean",
        # )
        # Normalize by library size
        nonzero_fraction = (target_counts > 0).float().mean()

        target_norm = target_counts / target_counts.sum(dim=1, keepdim=True) * 10_000#self.species_vocab_sizes[0]
        pred_norm = pred_counts / pred_counts.sum(dim=1, keepdim=True) * 10_000#self.species_vocab_sizes[0]
        
        count_loss_nb = negative_binomial_loss(
            pred=pred_norm.clamp(min=1e-6),
            target=target_norm,
            theta=reconstruction['theta'],
        )
        
        return self.recon_weight * count_loss_nb * (1 - nonzero_fraction)
    
    def _compute_kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2).clamp(max=100) - logvar.exp().clamp(max=100))
    
    def _compute_adversarial_loss(
        self,
        z: torch.Tensor,
        species_idx: int,
    ) -> torch.Tensor:
        # Detach z when training discriminator to prevent gradients flowing back
        z_detached = z.detach()
        
        # Discriminator loss (using detached z)
        species_logits_disc = self.discriminator(z_detached)
        target = torch.full((z.shape[0],), species_idx, dtype=torch.long, device=z.device)
        disc_loss = F.cross_entropy(species_logits_disc, target)
        
        # Generator loss (using non-detached z)
        species_logits_gen = self.discriminator(z)
        uniform_target = torch.ones_like(species_logits_gen) / self.n_species
        gen_loss = F.kl_div(
            F.log_softmax(species_logits_gen, dim=1),
            uniform_target,
            reduction='batchmean'
        )
        
        return disc_loss, self.adversarial_weight * gen_loss
    
    def _single_species_compute_loss(
        self, 
        batch: BatchData,
        outputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        target_species_id = batch.species_idx
        target_input = batch.data
        
        recon_loss = self._compute_reconstruction_loss(
            target_input,
            outputs['reconstructions'][target_species_id],
        )
        
        kl = self._compute_kl_loss(
            outputs['encoder_outputs'][target_species_id]['mu'],
            outputs['encoder_outputs'][target_species_id]['logvar']
        )
        
        # disc, adversarial = self._compute_adversarial_loss(
        #     outputs['encoder_outputs'][target_species_id]['z'],
        #     target_species_id,
        # )
        
        beta = self.get_current_beta()
        total_loss = recon_loss + kl * beta #+ adversarial + disc
        
        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl,
            "adversarial": torch.tensor(0.0, device=recon_loss.device),
        }

    def compute_loss(self, batch: BatchData, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._single_species_compute_loss(batch, outputs)
    
    def _single_species_forward(self, batch: BatchData):
        # Get target species ID for this batch
        target_species_id = batch.species_idx

        # 1. Direct reconstruction path
        target_encoder_outputs = self.encoders[str(target_species_id)](batch)
        
        # Decode to all species
        target_reconstruction = self.decoders[str(target_species_id)](target_encoder_outputs, batch.species_idx)
        
        return {
            'encoder_outputs': {target_species_id: target_encoder_outputs},
            'reconstructions': {target_species_id: target_reconstruction},
        }
    
    def forward(self, batch: BatchData):
        return self._single_species_forward(batch)

    
    @torch.no_grad()
    def get_latent_embeddings(
        self,
        species_data: Dict[str, Union[str, List[str], ad.AnnData, List[ad.AnnData]]],
        return_species: bool = True,
        batch_size: int = 512,
        device: Optional[torch.device] = "cuda",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get latent embeddings for cells in provided AnnData objects.
        
        Args:
            species_data: Dictionary mapping species names to AnnData objects
            return_species: Whether to return species labels
            batch_size: Batch size for processing
            device: Device to use for computation. If None, uses model's device
            
        Returns:
            If return_species:
                latents: Tensor of shape [n_cells, n_latent]
                species: Tensor of shape [n_cells] containing species indices
        """
        if device is None:
            device = next(self.parameters()).device
        elif device is not None:
            self.to(device)
            
        dataset = CrossSpeciesInferenceDataset(
            species_data=species_data,
            batch_size=batch_size,
        )
        
        # Get full dataset
        self.eval()  # Set to evaluation mode
        

        all_latents = []
        all_species = []
        
        for batch in dataset:
            # Move batch to device
            batch = BatchData(
                data=batch.data.to(device),
                species_idx=batch.species_idx,
            )
            
            # Get species-specific encoder            
            encoder_outputs = self.encoders[str(batch.species_idx)](batch)   

            all_latents.append(encoder_outputs['z'].cpu())
            
            if return_species:
                species_idx = torch.full((encoder_outputs['z'].shape[0],), batch.species_idx, dtype=torch.long, device=device)
                all_species.append(species_idx)
        
        # Concatenate all batches
        latents = torch.cat(all_latents, dim=0)

        if return_species:
            species = torch.cat(all_species, dim=0)
            return latents, species
        
        return latents


def negative_binomial_loss(pred, target, theta, eps=1e-8):
    """
    Negative binomial loss with learnable dispersion parameter theta.

    Args:
        pred: torch.Tensor, predicted mean parameter (mu).
        target: torch.Tensor, observed counts (y).
        theta: torch.Tensor, dispersion parameter (theta), must be positive.
        eps: float, small value for numerical stability.

    Returns:
        torch.Tensor: Scalar loss (mean negative log-likelihood).
    """
    # Ensure stability
    pred = pred.clamp(min=eps)
    theta = theta.clamp(min=eps)

    # Negative binomial log likelihood
    log_prob = (
        torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
        + theta * (torch.log(theta + eps) - torch.log(theta + pred + eps))
        + target * (torch.log(pred + eps) - torch.log(theta + pred + eps))
    )

    return -log_prob.mean()
