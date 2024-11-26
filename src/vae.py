from typing import Dict, Any, Union, List, Optional, Tuple
import anndata as ad
import pytorch_lightning as pl
import torch
from torch import cuda
import math
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from src.dataclasses import SparseExpressionData
from src.data import CrossSpeciesInferenceDataset

class GeneImportanceModule(nn.Module):
    def __init__(self, n_genes: int, n_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        
        self.global_weights = nn.Parameter(torch.ones(n_genes))

        # self.dropout = nn.Dropout(dropout)
        # self.context_net = nn.Sequential(
        #     nn.Linear(n_genes, n_hidden),
        #     nn.LayerNorm(n_hidden),
        #     nn.ReLU(),
        #     self.dropout,
        #     nn.Linear(n_hidden, n_genes),
        #     nn.Softplus()
        # )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_weights = F.softplus(self.global_weights)
        importance = global_weights
        return x * importance

    def get_l1_reg(self) -> torch.Tensor:
        """Calculate L1 regularization for gene importance weights."""
        return torch.mean(torch.abs(F.softplus(self.global_weights)))

class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        hidden_dims: list,
        n_context_hidden: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Gene importance module
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
        
        
        # Shared pathway (no species info)
        self.encoder = make_encoder_layers(n_genes, hidden_dims)
        self.mu = nn.Linear(hidden_dims[-1], n_latent)
        self.logvar = nn.Linear(hidden_dims[-1], n_latent)
        
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-20, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std if self.training else mu
        
    def forward(self, batch: SparseExpressionData) -> Dict[str, torch.Tensor]:
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
        
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)

        return {
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
   

class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        hidden_dims: list,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Create decoder layers
        layers = []
        dims = [n_latent] + hidden_dims + [n_genes]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]) if i < len(dims) - 2 else nn.Identity(),
                nn.ReLU() if i < len(dims) - 2 else nn.Softplus(),
                nn.Dropout(dropout_rate) if i < len(dims) - 2 else nn.Identity()
            ])
            
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z: Dict[str, torch.Tensor]) -> torch.Tensor:        
        return self.decoder_net(z['z'])

class SpeciesClassifier(nn.Module):
    def __init__(self, n_latent: int, n_species: int, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_species)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    def __init__(
        self,
        species_vocab_sizes: Dict[int, int],
        homology_edges: Dict[int, Dict[int, torch.Tensor]] | None = None,
        n_latent: int = 32,
        hidden_dims: list = [512, 256, 128],
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
        species_weight: float = 0.1,
        recon_weight: float = 1.0,
        homology_weight: float = 0.1,
        l1_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['homology_edges'])
        
        # Scale learning rate based on batch size
        if batch_size is not None:
            self.learning_rate = base_learning_rate * (batch_size / base_batch_size)
        else:
            self.learning_rate = base_learning_rate
            
        print(f"Scaled learning rate: {self.learning_rate:.2e}")
        
        self.recon_weight = recon_weight
        self.homology_weight = homology_weight
        self.l1_weight = l1_weight

        self.init_beta = init_beta
        self.final_beta = final_beta
        self.current_beta = init_beta  # Track current beta value
        
        # Store parameters
        self.species_vocab_sizes = species_vocab_sizes
        self.n_species = len(species_vocab_sizes)
        self.n_latent = n_latent

        # Create species-specific encoders/decoders
        self.encoders = nn.ModuleDict({
            str(species_id): Encoder(
                n_genes=vocab_size,
                n_latent=n_latent,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            )
            for species_id, vocab_size in species_vocab_sizes.items()
        })
    
        self.decoders = nn.ModuleDict({
            str(species_id): Decoder(
                n_genes=vocab_size,
                n_latent=n_latent,
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
                        torch.ones(len(edges), dtype=torch.float32)
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

        # Add species classifier
        self.species_classifier = SpeciesClassifier(
            n_latent=n_latent,
            n_species=len(species_vocab_sizes)
        )
        
        self.species_weight = species_weight

    
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

    def get_current_beta(self) -> float:
        """Calculate current beta value based on training progress"""
        if self.trainer is None:
            return self.init_beta
            
        # Get current epoch and total epochs
        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs
        
        # Linear warmup from init_beta to final_beta
        beta = self.init_beta + (self.final_beta - self.init_beta) * (current_epoch / max_epochs)
        return beta
    
    def validation_step(self, batch: SparseExpressionData, batch_idx: int):
        """Validation step."""
        # Forward pass
        predictions, encoder_outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, predictions, encoder_outputs)
        
        # Store for epoch end processing
        self.validation_step_outputs.append({
            "val_loss": loss_dict["loss"],
            "val_recon_loss": loss_dict["recon_loss"],
            "val_kl": loss_dict["kl"],
            "val_homology_loss": loss_dict["homology_loss"],
            "val_l1_reg": loss_dict["l1_reg"],
            "val_species_loss": loss_dict["species_loss"]
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
        encoder_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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
            target,
            log_input=False,
            full=True,
            reduction="mean"
        )
        
        kl = -0.5 * torch.mean(
            1 + encoder_outputs['logvar'] - 
            encoder_outputs['mu'].pow(2).clamp(max=100) - 
            encoder_outputs['logvar'].exp().clamp(max=100)
        )        
        
        homology_loss = torch.tensor(0.0, device=kl.device)
        if hasattr(self, "homology_edges") and hasattr(self, "homology_scores"):
            homology_loss = self.compute_homology_loss(predictions, batch)
        
        # Add L1 regularization for gene importance weights
        encoder = self.encoders[str(species_id)]
        l1_reg = encoder.gene_importance.get_l1_reg()
        
        # Use scheduled beta instead of learnable parameter
        current_beta = self.get_current_beta()
        self.current_beta = current_beta  # Store for logging
        
        # Add species classifier loss
        species_loss = self.compute_species_loss(encoder_outputs['z'], batch.species_idx)
        
        # Total loss with scheduled beta
        total_loss = (
            self.recon_weight * recon_loss + 
            current_beta * kl +
            self.homology_weight * homology_loss + 
            self.l1_weight * l1_reg +
            self.species_weight * species_loss
        )
        
        # Add L1 regularization to loss dict
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl": kl,
            "homology_loss": homology_loss,
            "l1_reg": l1_reg,
            "species_loss": species_loss
        }

    def training_step(self, batch: SparseExpressionData, batch_idx: int):
        # Forward pass
        predictions, encoder_outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, predictions, encoder_outputs)
        
        # Log metrics including current beta
        self.log("train_loss", loss_dict["loss"], sync_dist=True)
        self.log("train_recon_loss", loss_dict["recon_loss"], sync_dist=True)
        self.log("train_kl", loss_dict["kl"], sync_dist=True)
        self.log("train_homology_loss", loss_dict["homology_loss"], sync_dist=True)
        self.log("train_l1_reg", loss_dict["l1_reg"], sync_dist=True)
        self.log("beta", self.current_beta, sync_dist=True)
        self.log("train_species_loss", loss_dict["species_loss"], sync_dist=True)
        
        return loss_dict["loss"]

    # def compute_homology_loss(self, predictions: Dict[int, torch.Tensor], batch: SparseExpressionData) -> torch.Tensor:
    #     current_species_id = batch.species_idx[0].item()
    #     homology_loss = 0.0
    #     temperature = 0.1  # Temperature parameter
        
    #     for species_id in predictions:
    #         if species_id == current_species_id:
    #             continue
                
    #         edges = self.homology_edges[current_species_id][species_id]
    #         scores = F.softplus(self.homology_scores[str(current_species_id)][str(species_id)])
            
    #         src, dst = edges.t()
    #         src_pred = predictions[current_species_id][:, src]  # [batch_size, n_edges]
    #         dst_pred = predictions[species_id][:, dst]  # [batch_size, n_edges]
            
    #         # Normalize predictions along batch dimension
    #         src_norm = F.normalize(src_pred, dim=0)  # [batch_size, n_edges]
    #         dst_norm = F.normalize(dst_pred, dim=0)  # [batch_size, n_edges]
            
    #         # Compute positive similarities
    #         positive_sim = (src_norm * dst_norm).sum(dim=0) / temperature  # [n_edges]
            
    #         # Compute negative similarities using other edges as negatives
    #         # For each edge, use K random other edges as negatives
    #         K = 100  # number of negative samples per positive pair
    #         n_edges = len(edges)
            
    #         negative_loss = 0.0
    #         for i in range(0, n_edges, K):
    #             # Get random negative indices for this batch
    #             neg_indices = torch.randint(0, n_edges, (K,), device=src_pred.device)
                
    #             # Compute similarities with negative samples
    #             src_anchor = src_norm[:, i:i+1]  # [batch_size, 1]
    #             dst_negs = dst_norm[:, neg_indices]  # [batch_size, K]
                
    #             neg_sim = (src_anchor.transpose(0,1) @ dst_negs).squeeze(0) / temperature  # [K]
                
    #             # InfoNCE loss for this positive pair
    #             pos_term = positive_sim[i]
    #             neg_term = torch.logsumexp(neg_sim, dim=0)
    #             pair_loss = -(pos_term - neg_term)
                
    #             # Weight by homology score
    #             negative_loss += pair_loss * scores[i]
            
    #         pair_loss = negative_loss / n_edges
    #         homology_loss += pair_loss
                    
    #     return homology_loss / (len(predictions) - 1)
    
    def compute_homology_loss(self, predictions: Dict[int, torch.Tensor], batch: SparseExpressionData) -> torch.Tensor:
        current_species_id = batch.species_idx[0].item()
        homology_loss = 0.0
        
        for species_id in predictions:
            if species_id == current_species_id:
                continue
                
            # Get edges and scores for this species pair
            edges = self.homology_edges[current_species_id][species_id]
            scores = F.softplus(self.homology_scores[str(current_species_id)][str(species_id)])
            
            src, dst = edges.t()
            src_pred = predictions[current_species_id][:, src]
            dst_pred = predictions[species_id][:, dst]
            
            # Center the predictions
            src_centered = src_pred - src_pred.mean(dim=0)
            dst_centered = dst_pred - dst_pred.mean(dim=0)
            
            # Compute correlation
            covariance = (src_centered * dst_centered).mean(dim=0)
            src_std = src_centered.std(dim=0)
            dst_std = dst_centered.std(dim=0)
            correlation = covariance / (src_std * dst_std + 1e-8)
            
            # Weight each edge's contribution by its learnable score
            edge_loss = (1 - correlation) * scores
            pair_loss = edge_loss.mean()
            
            homology_loss += pair_loss
        
        return homology_loss / (len(predictions) - 1)  # Average across species pairs

    def compute_species_loss(self, z: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        """Compute maximum entropy loss for species prediction."""
        species_logits = self.species_classifier(z.detach())  # Detach for classifier training
        
        # Apply temperature scaling to make logits more extreme
        temperature = 10.0
        species_logits = species_logits * temperature
        
        # Compute probabilities
        log_probs = F.log_softmax(species_logits, dim=1)
        
        # Uniform distribution
        uniform = torch.ones_like(species_logits) / self.n_species
        
        # Reverse KL divergence (tends to force more uniform distribution)
        entropy_loss = torch.sum(uniform * (torch.log(uniform + 1e-8) - log_probs), dim=1).mean()
        
        return entropy_loss

    
    def forward(self, batch: SparseExpressionData):
        # Get species ID for this batch
        species_id = batch.species_idx[0].item()
        
        # Forward pass through encoder
        encoder = self.encoders[str(species_id)]
        encoder_outputs = encoder(batch)
        
        # Get predictions for all species
        predictions = {}
        for decoder_species_id in self.decoders:
            predictions[int(decoder_species_id)] = self.decoders[decoder_species_id](encoder_outputs)
        
        return predictions, encoder_outputs

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
            encoder_outputs = self.encoders[str(species_id)](batch)    
            all_latents.append(encoder_outputs['z'].cpu())
            
            if return_species:
                all_species.append(batch.species_idx)
        
        # Concatenate all batches
        latents = torch.cat(all_latents, dim=0)

        if return_species:
            species = torch.cat(all_species, dim=0)
            return latents, species
        
        return latents

class DivergenceEarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = 'loss',
        divergence_threshold: float = 0.05,  # Maximum allowed gap between train/val
        check_divergence_after: int = 500,  # Start checking after this many steps
        verbose: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.divergence_threshold = divergence_threshold
        self.check_divergence_after = check_divergence_after
        self.verbose = verbose
        
    def _check_divergence(self, trainer):
        # Get current training and validation metrics
        train_loss = trainer.callback_metrics.get(f'train_{self.monitor}')
        val_loss = trainer.callback_metrics.get(f'val_{self.monitor}')
        
        print(train_loss, val_loss)
        if train_loss is None or val_loss is None:
            return False
            
        # Convert to float values
        train_loss = train_loss.item()
        val_loss = val_loss.item()
        
        # Calculate absolute relative gap
        gap = (val_loss - train_loss) / val_loss
        

        # Check if we're past the initial training period
        if trainer.global_step < self.check_divergence_after:
            return False
        
        if gap > self.divergence_threshold and self.verbose:
            print(f"Divergence detected: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, gap={gap:.4f}")
            
        return gap > self.divergence_threshold
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if self._check_divergence(trainer):
            if self.verbose:
                print(f"Stopping training due to divergence at epoch {trainer.current_epoch}")
            trainer.should_stop = True