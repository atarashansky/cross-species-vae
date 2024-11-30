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
    def __init__(self, n_genes: int, n_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.global_weights = nn.Parameter(torch.ones(n_genes))

        self.dropout = nn.Dropout(dropout)
        self.context_net = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            self.dropout,
            nn.Linear(n_hidden, n_genes),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_weights = F.softplus(self.global_weights)
        importance = global_weights * self.context_net(x)
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
        
        self.input_bn = nn.BatchNorm1d(n_genes)
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
        
    def forward(self, x: Union[SparseExpressionData, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Handle sparse input
        if isinstance(x, SparseExpressionData):
            dense_x = torch.zeros(
                x.batch_size,
                x.n_genes,
                device=x.values.device,
                dtype=x.values.dtype
            )
            dense_x[x.batch_idx, x.gene_idx] = x.values
        else:
            dense_x = x
        
        # Calculate library size for normalization
        lib_size = dense_x.sum(dim=1, keepdim=True)
        normalized_x = dense_x / (lib_size + 1e-6)
        
        x = self.input_bn(normalized_x)
        x = self.gene_importance(x)
        
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
        
        # Technical scaling network
        self.scaling_net = nn.Sequential(
            nn.Linear(n_latent + species_embedding.embedding_dim, n_genes),
            nn.Softplus()  # Ensure positive scaling
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
    
    def forward(self, z: Dict[str, torch.Tensor], species_idx: torch.Tensor) -> torch.Tensor:
        # Get biological factors
        species_embedding = self.species_embedding(species_idx)
        z_input = torch.cat([z['z'], species_embedding], dim=1)
        bio_factors = self.decoder_net(z_input)
        
        # Get technical scaling factors
        scaling_factors = self.scaling_net(z_input)

        # Combine all factors
        return bio_factors * scaling_factors

class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    def __init__(
        self,
        species_vocab_sizes: Dict[int, int],
        homology_edges: Dict[int, Dict[int, torch.Tensor]] | None = None,
        homology_scores: Dict[int, Dict[int, torch.Tensor]] | None = None,
        n_latent: int = 32,
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.1,
        base_learning_rate: float = 1e-3,
        base_batch_size: int = 32,
        batch_size: int | None = None,
        min_learning_rate: float = 1e-5,
        species_embedding_dim: int = 32,
        warmup_epochs: float = 0.1,
        init_beta: float = 1e-3,
        final_beta: float = 0.1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        recon_weight: float = 1.0,
        similarity_weight: float = 1.0,
        multi_species_mode: bool = False,
        learn_homology_weights: bool = False,
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
        self.similarity_weight = similarity_weight

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
        
        self.multi_species_mode = multi_species_mode
        self.learn_homology_weights = learn_homology_weights

    
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
    

    def training_step(self, batch: SparseExpressionData, batch_idx: int):
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"], sync_dist=True)
        self.log("train_recon", loss_dict["recon"], sync_dist=True)
        self.log("train_kl", loss_dict["kl"], sync_dist=True)
        self.log("train_similarity", loss_dict["similarity"], sync_dist=True)
        
        return loss_dict["loss"]
        
    def validation_step(self, batch: SparseExpressionData, batch_idx: int):
        """Validation step."""
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        
        # Store for epoch end processing
        self.validation_step_outputs.append({
            "val_loss": loss_dict["loss"],
            "val_recon": loss_dict["recon"],
            "val_kl": loss_dict["kl"],
            "val_similarity": loss_dict["similarity"],
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
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        return F.poisson_nll_loss(reconstruction.clamp(min=1e-6), target_input, log_input=False, full=True, reduction="mean")
    
    def _compute_kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2).clamp(max=100) - logvar.exp().clamp(max=100))
    
    def _single_species_compute_loss(
        self, 
        batch: SparseExpressionData,
        outputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        target_species_id = batch.species_idx[0].item()
        target_input = outputs['inputs'][target_species_id]
        
        current_beta = self.get_current_beta()
        
                
        recon_loss = self._compute_reconstruction_loss(
            target_input,
            outputs['reconstructions'][target_species_id]
        )
        
        kl = self._compute_kl_loss(
            outputs['encoder_outputs'][target_species_id]['mu'],
            outputs['encoder_outputs'][target_species_id]['logvar']
        )
        
        total_loss = (
            self.recon_weight * recon_loss +
            current_beta * kl
        )
        
        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl,
            "similarity": torch.tensor(0.0, device=recon_loss.device),
        }
    
    def _multi_species_compute_loss(self, batch: Dict[int, SparseExpressionData], outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        # 1. Within-species reconstruction & KL losses (reduced weight)
        recon_loss = 0
        kl_loss = 0
        for species_id in batch:
            # Reconstruction loss
            recon = outputs['reconstructions'][species_id]
            target_input = outputs['inputs'][species_id]
            recon_loss += self._compute_reconstruction_loss(target_input, recon)
            
            # KL loss
            encoder_out = outputs['encoder_outputs'][species_id]
            kl_loss += self._compute_kl_loss(encoder_out['mu'], encoder_out['logvar'])
        
        # Average across species
        n_species = len(batch)
        recon_loss /= n_species
        kl_loss /= n_species
        
        # 2. Cross-species similarity losses
        similarity_loss = 0
        n_pairs = 0
        
        # Compare all pairs of species
        species_ids = list(batch.keys())
        for i in range(len(species_ids)):
            for j in range(i + 1, len(species_ids)):
                species1_id = species_ids[i]
                species2_id = species_ids[j]
                
                # Get latent representations
                z1 = outputs['encoder_outputs'][species1_id]['z']
                z2 = outputs['encoder_outputs'][species2_id]['z']
                
                # Compute similarities
                sim = self.compute_similarity(z1, z2)
                homology = self.compute_homology_agreement(
                    species1_id, 
                    species2_id,
                    outputs
                )
                
                # MSE between similarity and homology
                pair_loss = F.mse_loss(sim, homology)
                similarity_loss += pair_loss
                n_pairs += 1
        
        similarity_loss /= max(n_pairs, 1)  # Average across pairs
        
        # Combine losses with weights
        current_beta = self.get_current_beta()

        losses['recon'] = recon_loss * self.recon_weight
        losses['kl'] = kl_loss * current_beta
        losses['similarity'] = similarity_loss * self.similarity_weight
        
        # Total loss
        losses['loss'] = (
            losses['recon'] + 
            losses['kl'] + 
            losses['similarity']
        )
        
        return losses
    
    def compute_loss(self, batch: Union[SparseExpressionData, Dict[int, SparseExpressionData]], outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.multi_species_mode:
            return self._multi_species_compute_loss(batch, outputs)
        else:
            return self._single_species_compute_loss(batch, outputs)
    
    def _single_species_forward(self, batch: SparseExpressionData):
        """
        Forward pass that computes:
        1. Direct reconstruction: Input A → Encoder A → Decoder A
        2. Cross-species reconstructions: Synthetic B/C → Encoder B/C → Decoder A/B/C
        
        Returns:
            Dict containing:
            - direct: Dict with encoder outputs and reconstructions for all decoders
            - cross_species: Dict mapping species_id to their encoder outputs and reconstructions
        """
        # Get target species ID for this batch
        target_species_id = batch.species_idx[0].item()
        
        # Create dense target tensor
        target_input = torch.zeros(
            batch.batch_size,
            batch.n_genes,
            device=batch.values.device,
            dtype=batch.values.dtype
        )
        target_input[batch.batch_idx, batch.gene_idx] = batch.values
        
        # 1. Direct reconstruction path
        target_encoder_outputs = self.encoders[str(target_species_id)](batch)
        
        # Decode to all species
        target_reconstruction = self.decoders[str(target_species_id)](target_encoder_outputs, batch.species_idx)
        
        return {
            'encoder_outputs': {target_species_id: target_encoder_outputs},
            'reconstructions': {target_species_id: target_reconstruction},
            'inputs': {target_species_id: target_input}
        }
    
    def _multi_species_forward(self, batch: Dict[int, SparseExpressionData]):
        """
        Forward pass that computes:
        1. Direct reconstruction: Input A → Encoder A → Decoder A
        2. Cross-species reconstructions: Synthetic B/C → Encoder B/C → Decoder A/B/C
        
        Returns:
            Dict containing:
            - direct: Dict with encoder outputs and reconstructions for all decoders
            - cross_species: Dict mapping species_id to their encoder outputs and reconstructions
        """
        # Get target species ID for this batch
        encoder_outputs = {}
        reconstructions = {}
        inputs = {}
        
        # Encode each species batch
        for species_id, species_batch in batch.items():
            encoder_outputs[species_id] = self.encoders[str(species_id)](species_batch)
            
            reconstructions[species_id] = self.decoders[str(species_id)](
                encoder_outputs[species_id], 
                species_batch.species_idx
            )

            target_input = torch.zeros(
                species_batch.batch_size,
                species_batch.n_genes,
                device=species_batch.values.device,
                dtype=species_batch.values.dtype
            )
            target_input[species_batch.batch_idx, species_batch.gene_idx] = species_batch.values
            
            inputs[species_id] = target_input
        
        return {
            'encoder_outputs': encoder_outputs,
            'reconstructions': reconstructions,
            'inputs': inputs,
        }   
    

    def forward(self, batch: Union[SparseExpressionData, Dict[int, SparseExpressionData]]):
        if self.multi_species_mode:
            return self._multi_species_forward(batch)
        else:
            return self._single_species_forward(batch)

    def compute_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between latent vectors."""
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        return torch.sum(z1_norm * z2_norm, dim=1)
    
    def compute_homology_agreement(
        self, 
        species1_id: int,
        species2_id: int,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute homology agreement between batches of cells from two species."""
        edges = self.homology_edges[species1_id][species2_id]
        weights = F.softplus(self.homology_scores[str(species1_id)][str(species2_id)])
        
        if not self.learn_homology_weights:
            weights = weights.detach()
        
        expr1 = outputs['inputs'][species1_id]
        expr2 = outputs['inputs'][species2_id]
        
        expr1_homolog = expr1[:, edges[:, 0]] * weights[None, :]
        expr2_homolog = expr2[:, edges[:, 1]] * weights[None, :]
        
        expr1_centered = expr1_homolog - expr1_homolog.mean(dim=1, keepdim=True)
        expr2_centered = expr2_homolog - expr2_homolog.mean(dim=1, keepdim=True)
        
        std1 = expr1_centered.std(dim=1, keepdim=True)
        std2 = expr2_centered.std(dim=1, keepdim=True)
        correlations = (expr1_centered * expr2_centered).mean(dim=1) / (std1 * std2 + 1e-8)

        return correlations
    
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