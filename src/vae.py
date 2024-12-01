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
        n_species: int,
        n_latent: int,
        species_embedding: nn.Embedding,
        hidden_dims: list,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.log_theta = nn.Parameter(torch.ones(n_genes) * 2.3)

        # Technical scaling network
        self.scaling_net_single = nn.Sequential(
            nn.Linear(n_latent, n_genes),
            nn.Sigmoid()
        )

        self.scaling_net_fusion = nn.Sequential(
            nn.Linear(n_latent * n_species, n_genes),
            nn.Sigmoid()
        )
        
        # Biological decoder network
        layers = []
        dims = [n_latent] + hidden_dims + [n_genes]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]) if i < len(dims) - 2 else nn.Identity(),
                nn.ReLU() if i < len(dims) - 2 else nn.Softplus(),
                nn.Dropout(dropout_rate) if i < len(dims) - 2 else nn.Identity()
            ])
            
        # self.species_embedding = species_embedding
        self.decoder_net_single = nn.Sequential(*layers)

        layers = []
        dims = [n_latent * n_species] + hidden_dims + [n_genes]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]) if i < len(dims) - 2 else nn.Identity(),
                nn.ReLU() if i < len(dims) - 2 else nn.Softplus(),
                nn.Dropout(dropout_rate) if i < len(dims) - 2 else nn.Identity()
            ])
            
        self.decoder_net_fusion = nn.Sequential(*layers)        
    
    def forward(self, z: Dict[str, torch.Tensor], fusion: bool = False) -> Dict[str, torch.Tensor]:
        if fusion:
            return self._fusion_forward(z)
        else:
            return self._single_forward(z)
    
    def _single_forward(self, z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get biological factors
        # batch_species_idx = torch.full((z['z'].shape[0],), species_idx, dtype=torch.long, device=z['z'].device)
        # species_embedding = self.species_embedding(batch_species_idx)
        # z_input = torch.cat([z['z'], species_embedding], dim=1)
        bio_factors = self.decoder_net_single(z['z'])
        
        # Get technical scaling factors
        scaling_factors = self.scaling_net_single(z['z'])

        # Combine all factors
        mean = bio_factors * scaling_factors
        theta = torch.exp(self.log_theta)  # Ensure theta is positive
        
        return {
            'mean': mean,
            'theta': theta
        }

    def _fusion_forward(self, z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # Get biological factors
        # batch_species_idx = torch.full((z['z'].shape[0],), species_idx, dtype=torch.long, device=z['z'].device)
        # # species_embedding = self.species_embedding(batch_species_idx)
        # z_input = torch.cat([z['z'], species_embedding], dim=1)
        bio_factors = self.decoder_net_fusion(z['z'])
        
        # Get technical scaling factors
        scaling_factors = self.scaling_net_fusion(z['z'])

        # Combine all factors
        mean = bio_factors * scaling_factors
        theta = torch.exp(self.log_theta)  # Ensure theta is positive
        
        return {
            'mean': mean,
            'theta': theta
        }        

class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    STAGE_MAPPING = {
        0: "direct_recon",
        1: "transform_recon",
        2: "homology_loss",
    }

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
        homology_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['homology_edges', 'homology_scores'])
        
        # Scale learning rate based on batch size
        if batch_size is not None:
            self.learning_rate = base_learning_rate * (batch_size / base_batch_size)
        else:
            self.learning_rate = base_learning_rate
                    
        self.recon_weight = recon_weight
        self.homology_weight = homology_weight

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
                n_species=self.n_species,
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
        self.log("train_homology", loss_dict["homology"].detach(), sync_dist=True)
        
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
            "val_homology": loss_dict["homology"].detach(),
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

        print(f"Current epoch: {self.trainer.current_epoch}", "Current stage: ", self.get_stage())
        
        
   
    def _compute_homology_loss(self, outputs: Dict[int, Any], batch: BatchData) -> torch.Tensor:   
        if self.get_stage() != "homology_loss":
            return torch.tensor(0.0, device=self.device)
                        
        current_species_id = batch.species_idx
        homology_loss = 0.0
        
        # Get encoder outputs for current species
        encoder_outputs = outputs['encoder_outputs'][current_species_id]
        reconstructions = outputs['reconstructions'][current_species_id]
        
        # Get decoder outputs for all species
        all_species_outputs = {current_species_id: reconstructions['mean']}
        for species_id in self.species_vocab_sizes.keys():
            if species_id == current_species_id:
                continue
            
            decoder_output = self.decoders[str(species_id)](encoder_outputs, fusion=False)
            all_species_outputs[species_id] = decoder_output['mean']

        # Compute homology loss across species pairs
        for species_id in self.species_vocab_sizes.keys():
            if species_id == current_species_id:
                continue

            # Get edges and scores for this species pair
            edges = self.homology_edges[current_species_id][species_id]
            scores = torch.sigmoid(self.homology_scores[str(current_species_id)][str(species_id)])

            src, dst = edges.t()
            src_pred = all_species_outputs[current_species_id][:, src]
            dst_pred = all_species_outputs[species_id][:, dst]

            # Center the predictions
            src_centered = src_pred - src_pred.mean(dim=0)
            dst_centered = dst_pred - dst_pred.mean(dim=0)

            # Compute correlation
            covariance = (src_centered * dst_centered).mean(dim=0)
            src_std = src_centered.std(dim=0)
            dst_std = dst_centered.std(dim=0)
            correlation = covariance / (src_std * dst_std + 1e-8)

            alignment_loss = -torch.mean(correlation * scores)

            homology_loss += alignment_loss

        return self.homology_weight * homology_loss / (len(self.species_vocab_sizes) - 1)

    def _compute_reconstruction_loss(
        self,
        outputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        if self.get_stage() == "homology_loss":
            return torch.tensor(0.0, device=self.device)
        
        reconstruction = outputs['reconstructions']

        count_loss_nb = torch.tensor(0.0, device=self.device)
        for species_id, input in outputs['inputs'].items():
            
            target_counts = torch.exp(input) - 1
            pred_counts = torch.exp(reconstruction[species_id]['mean']) - 1
            nonzero_fraction = (target_counts > 0).float().mean()

            target_norm = target_counts / target_counts.sum(dim=1, keepdim=True) * 10_000
            pred_norm = pred_counts / pred_counts.sum(dim=1, keepdim=True) * 10_000
            
            count_loss_nb += negative_binomial_loss(
                pred=pred_norm.clamp(min=1e-6),
                target=target_norm,
                theta=reconstruction[species_id]['theta'],
            )
    
        return self.recon_weight * count_loss_nb * (1 - nonzero_fraction)
    
    def _compute_kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if self.get_stage() == "homology_loss":
            return torch.tensor(0.0, device=self.device)
        
        return -0.5 * torch.mean(1 + logvar - mu.pow(2).clamp(max=100) - logvar.exp().clamp(max=100))
    
    def _single_species_compute_loss(
        self, 
        batch: BatchData,
        outputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        target_species_id = batch.species_idx

        recon_loss = self._compute_reconstruction_loss(outputs)
        
        kl = self._compute_kl_loss(
            outputs['encoder_outputs'][target_species_id]['mu'],
            outputs['encoder_outputs'][target_species_id]['logvar']
        )

        homology_loss = self._compute_homology_loss(outputs, batch)
        
        beta = self.get_current_beta()
        total_loss = recon_loss + kl * beta + homology_loss
        
        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl,
            "homology": homology_loss,
        }

    def compute_loss(self, batch: BatchData, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._single_species_compute_loss(batch, outputs)
    
    def _transform_expression(
        self,
        batch: BatchData,
        src_species: int,
        dst_species: int,
    ) -> torch.Tensor:
        # Create dense transformation matrix
        edges = self.homology_edges[src_species][dst_species]
        scores = torch.sigmoid(self.homology_scores[str(src_species)][str(dst_species)])
        
        n_src_genes = self.species_vocab_sizes[src_species]
        n_dst_genes = self.species_vocab_sizes[dst_species]
        transform_matrix = torch.zeros(
            n_dst_genes,
            n_src_genes,
            device=batch.data.device
        )
        transform_matrix[edges[:, 1], edges[:, 0]] = scores
        # Normalize transform matrix so src edges sum to 1.0 per dst gene
        transform_matrix = transform_matrix / (transform_matrix.sum(dim=1, keepdim=True) + 1e-8)

        # Create dense input matrix
        x = batch.data

        # Transform expression data
        transformed = x @ transform_matrix.t()
        return transformed
        
    def _single_species_forward(self, batch: BatchData):
        # Get target species ID for this batch
        target_species_id = batch.species_idx

        # 1. Direct reconstruction path
        target_encoder_outputs = self.encoders[str(target_species_id)](batch)
        
        # Decode to all species
        encoder_outputs = {target_species_id: target_encoder_outputs}
        reconstructions = {}
        inputs = {target_species_id: batch.data}

        if self.get_stage() == "transform_recon":
            # 2. Transform expression to other species
            for species_id in self.species_vocab_sizes.keys():
                if species_id == target_species_id:
                    continue
                
                transformed = self._transform_expression(batch, target_species_id, species_id)
                inputs[species_id] = transformed
                transformed_encoder_outputs = self.encoders[str(species_id)](transformed)                
                encoder_outputs[species_id] = transformed_encoder_outputs
            
            encoder_outputs_concat = {'z': torch.cat([encoder_outputs[species_id]['z'] for species_id in self.species_vocab_sizes.keys()], dim=1)}
            for species_id in self.species_vocab_sizes.keys():                
                transformed_reconstruction = self.decoders[str(species_id)](encoder_outputs_concat, fusion=True)
                reconstructions[species_id] = transformed_reconstruction
        else:
            target_reconstruction = self.decoders[str(target_species_id)](target_encoder_outputs, fusion=False)
            reconstructions = {target_species_id: target_reconstruction}

        return {
            'encoder_outputs': encoder_outputs,
            'reconstructions': reconstructions,
            'inputs': inputs,
        }
    
    def forward(self, batch: BatchData):
        return self._single_species_forward(batch)

    def get_stage(self): # 0: direct recon loss, 1: transform recon loss with fused latents, 2: homology loss
        return self.STAGE_MAPPING[self.trainer.current_epoch % 3]
    
    @torch.no_grad()
    def get_latent_embeddings(
        self,
        species_data: Dict[str, Union[str, List[str], ad.AnnData, List[ad.AnnData]]],
        return_species: bool = True,
        batch_size: int = 512,
        fusion: bool = False,
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
                    
            if fusion:
                # Get latents for all species
                concat_latents = [
                    self.encoders[str(species_id)](
                        self._transform_expression(batch, batch.species_idx, species_id) 
                        if species_id != batch.species_idx 
                        else batch
                    )['z'].cpu()
                    for species_id in self.species_vocab_sizes.keys()
                ]
            else:
                # Get latents only for input species
                concat_latents = [self.encoders[str(batch.species_idx)](batch)['z'].cpu()]

            latents = torch.cat(concat_latents, dim=1)
            all_latents.append(latents)
            
            if return_species:
                species_idx = torch.full((latents.shape[0],), batch.species_idx, dtype=torch.long, device=device)
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
