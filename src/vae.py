import math
from typing import Dict, Any

import pytorch_lightning as pl
import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F

from src.dataclasses import SparseExpressionData, EncoderOutput, SpeciesLatents, LossOutput



class Encoder(nn.Module):
    """Multi-scale encoder with species-specific and global latent spaces."""

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_latent: int,
        hidden_dims: list,
        dropout_rate: float = 0.1,
        edge_dim: int = 1,
        species_dim: int = 32,
    ):
        super().__init__()
        self.n_genes = n_genes

        # Species embedding
        self.species_embedding = nn.Embedding(n_species, species_dim)

        # Input projection
        self.input_proj = nn.Linear(1 + species_dim, hidden_dims[0])

        # Multi-scale GNN layers with learnable edge weights
        self.gnn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        hidden_dims[i], hidden_dims[i + 1]
                    ),  # Node feature transformation
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
                for i in range(len(hidden_dims) - 1)
            ]
        )

        # Species-specific latent projectors
        total_hidden = sum(hidden_dims)
        self.species_mu = nn.ModuleList(
            [nn.Linear(total_hidden, n_latent) for _ in range(n_species)]
        )
        self.species_var = nn.ModuleList(
            [nn.Linear(total_hidden, n_latent) for _ in range(n_species)]
        )

        # Global latent projectors
        self.global_mu = nn.Linear(total_hidden, n_latent)
        self.global_var = nn.Linear(total_hidden, n_latent)

    def message_passing(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        gene_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Message passing using efficient index_add_ operations."""
        # Normalize edge weights
        edge_weights = F.softmax(edge_weights, dim=0)

        # Create output tensor
        out = torch.zeros_like(h)
        src, dst = edge_index

        # First, compute average hidden states per gene
        gene_h = torch.zeros(self.n_genes, h.size(1), device=h.device, dtype=h.dtype)
        gene_counts = torch.zeros(self.n_genes, 1, device=h.device, dtype=h.dtype)

        # Ensure gene_idx is within bounds
        if torch.any(gene_idx >= self.n_genes):
            raise ValueError(f"gene_idx contains indices >= n_genes ({self.n_genes})")
        if torch.any(gene_idx < 0):
            raise ValueError("gene_idx contains negative indices")

        # Ensure src/dst indices are within bounds
        if torch.any(src >= self.n_genes) or torch.any(dst >= self.n_genes):
            raise ValueError(f"edge_index contains indices >= n_genes ({self.n_genes})")
        if torch.any(src < 0) or torch.any(dst < 0):
            raise ValueError("edge_index contains negative indices")

        # Compute gene-level representations
        gene_h.index_add_(0, gene_idx, h)
        gene_counts.index_add_(0, gene_idx, torch.ones_like(h[:, :1]))
        gene_h = gene_h / gene_counts.clamp(min=1)

        # Compute weighted messages between genes
        messages = gene_h[dst] * edge_weights.to(h.dtype).unsqueeze(-1)

        # Aggregate messages per gene
        gene_messages = torch.zeros_like(gene_h)
        gene_messages.index_add_(0, src, messages)

        # Distribute messages back to all cells for each gene
        out = gene_messages[gene_idx]

        return out
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        batch: SparseExpressionData,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> EncoderOutput:
        # Map species_idx to match sparse values using batch_idx
        expanded_species_idx = batch.species_idx[batch.batch_idx]
        
        # Get species embeddings and ensure correct shape
        species_emb = self.species_embedding(expanded_species_idx)
        x = batch.values.view(-1, 1)

        # Combine expression and species info
        h = torch.cat([x, species_emb], dim=-1)
        h = self.input_proj(h)

        # Ensure edge_index has correct shape and dtype
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        edge_index = edge_index.long()

        # Convert edge attributes to match hidden states dtype
        edge_weights = edge_attr.squeeze(-1).to(h.dtype)

        # Multi-scale feature extraction
        features = [h]
        for gnn in self.gnn_layers:
            # Message passing with edge weights
            msg = self.message_passing(h, edge_index, edge_weights, batch.gene_idx)

            # Update node features
            h = gnn(h + msg)
            features.append(h)

        # Combine features from all scales
        multi_scale = torch.cat(features, dim=1)

        # Generate species-specific latent distributions
        species_latents = {}
        unique_species = torch.unique(batch.species_idx)
        for species in unique_species:
            species_mask = expanded_species_idx == species
            if species_mask.any():
                species_features = multi_scale[species_mask]
                mu = self.species_mu[species.item()](species_features)
                logvar = self.species_var[species.item()](species_features)
                z = self.reparameterize(mu, logvar)
                species_latents[species.item()] = SpeciesLatents(latent=z, species_mask=species_mask, mu=mu, logvar=logvar)

        # Generate global latent distribution (per-batch)
        # First, aggregate features per batch
        batch_features = torch.zeros(
            batch.batch_size,
            multi_scale.size(1),
            device=multi_scale.device,
            dtype=multi_scale.dtype,
        )
        batch_counts = torch.zeros(
            batch.batch_size,
            1,
            device=multi_scale.device,
            dtype=multi_scale.dtype,
        )
        batch_features.index_add_(0, batch.batch_idx, multi_scale)
        batch_counts.index_add_(0, batch.batch_idx, torch.ones_like(multi_scale[:, :1]))
        batch_features = batch_features / batch_counts.clamp(min=1)

        global_mu = self.global_mu(batch_features)
        global_var = self.global_var(batch_features)
        global_latent = self.reparameterize(global_mu, global_var)
        return EncoderOutput(species_latents = species_latents, global_latent = global_latent, global_mu=global_mu, global_logvar = global_var)


class Decoder(nn.Module):
    """Multi-scale decoder with species-specific components."""

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_latent: int,
        hidden_dims: list,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Species-specific decoders
        self.species_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_latent * 2, hidden_dims[0]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    *[
                        nn.Sequential(
                            nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate),
                        )
                        for i in range(len(hidden_dims) - 1)
                    ],
                    nn.Linear(hidden_dims[-1], n_genes),
                )
                for _ in range(n_species)
            ]
        )
        # Separate softplus activation to apply only to non-zero genes
        self.activation = nn.Softplus()

    def forward(
        self,
        encoded: EncoderOutput,
        batch: SparseExpressionData,
    ) -> torch.Tensor:
        """Decode latent vectors back to gene expression space.

        Args:
            species_latents: Dict mapping species ID to (latent_vector, species_mask)
            global_latent: Global latent vector per batch
            batch: SparseExpressionData containing batch indices, gene indices, and species indices
        """
        # Initialize output tensor to match input size
        output = torch.zeros_like(batch.values)
        
        # Process each species
        for i in range(len(self.species_decoders)):
            decoder = self.species_decoders[i]
            if i in encoded.species_latents:
                # Get latent vector and batch mapping for this species
                species_latent = encoded.species_latents[i] 
                z, species_mask = species_latent.latent, species_latent.species_mask

                # Get corresponding batch indices
                batch_species_idx = batch.batch_idx[species_mask]
                gene_species_idx = batch.gene_idx[species_mask]

                # Get corresponding global latents
                expanded_global = encoded.global_latent[batch_species_idx]
                combined_z = torch.cat([z, expanded_global], dim=1)
                
                # Decode to full gene space
                decoded = decoder(combined_z)
                
                species_output = decoded[batch_species_idx, gene_species_idx]
                
                # Apply softplus activation
                species_output = self.activation(species_output)
                
                # Place the outputs in the correct positions in the output tensor
                output[species_mask] = species_output
        
        return output


class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        homology_edges: torch.Tensor,
        homology_scores: torch.Tensor,
        n_latent: int = 32,
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.1,
        species_dim: int = 32,
        l1_lambda: float = 0.01,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        warmup_epochs: float = 0.1,
        num_nodes: int = 1,
        num_gpus_per_node: int = 1,
        gradient_accumulation_steps: int = 1,
        batch_size: int = 128,
        temperature: float = 0.1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
    ):
        """
        Initialize VAE model.

        Args:
            n_genes: Number of genes in vocabulary
            n_species: Number of species
            n_latent: Dimension of latent space
            hidden_dims: List of hidden dimensions for encoder/decoder
            dropout_rate: Dropout rate
            homology_edges: Tensor of homology edges
            homology_scores: Tensor of homology edge scores
            species_dim: Dimension of species embedding
            l1_lambda: L1 regularization weight
            learning_rate: Initial learning rate
            min_learning_rate: Minimum learning rate for scheduler
            warmup_epochs: Fraction of epoch for warmup steps
            num_nodes: Number of compute nodes
            num_gpus_per_node: Number of GPUs per node
            gradient_accumulation_steps: Number of gradient accumulation steps
            batch_size: Batch size per GPU
            temperature: Temperature for sampling
            gradient_clip_val: Maximum gradient norm
            gradient_clip_algorithm: Algorithm for gradient clipping
        """
        super().__init__()
        self.save_hyperparameters()

        # Store model parameters
        self.n_genes = n_genes
        self.n_species = n_species
        self.n_latent = n_latent
        self.l1_lambda = l1_lambda
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Training configuration
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.world_size = num_nodes * num_gpus_per_node
        self.global_batch_size = (
            batch_size * self.world_size * gradient_accumulation_steps
        )

        # Register homology information with correct dtype
        self.register_buffer("homology_edges", homology_edges)
        self.register_buffer(
            "homology_scores", homology_scores.float()
        )  # Convert to float

        # Initialize encoder and decoder
        self.encoder = Encoder(
            n_genes=n_genes,
            n_species=n_species,
            n_latent=n_latent,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            species_dim=species_dim,
        )

        self.decoder = Decoder(
            n_genes=n_genes,
            n_species=n_species,
            n_latent=n_latent,
            hidden_dims=hidden_dims[::-1],  # Reverse hidden dims for decoder
            dropout_rate=dropout_rate,
        )

        # Initialize validation metrics
        self.validation_step_outputs = []

        # Add memory tracking attributes
        self.prev_gpu_memory = 0
        self.max_gpu_memory = 0

        # Store learning rate scheduler parameters
        self.min_learning_rate = min_learning_rate
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = None

    def _log_memory_stats(self, step_type: str, batch_size: int):
        """Log current GPU memory usage."""
        if cuda.is_available():
            # Current memory in GB
            current = cuda.memory_reserved() / 1024**3
            # Maximum memory in GB
            max_memory = cuda.max_memory_reserved() / 1024**3
            # Memory increase since last step
            memory_increase = current - self.prev_gpu_memory

            # Update tracking
            self.prev_gpu_memory = current
            self.max_gpu_memory = max(self.max_gpu_memory, max_memory)

            # Log metrics
            self.log(f"{step_type}_gpu_memory", current, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log(f"{step_type}_gpu_memory_peak", max_memory, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log(f"{step_type}_gpu_memory_increase", memory_increase, prog_bar=True, batch_size=batch_size, sync_dist=True)

    def encode(
        self, batch: SparseExpressionData
    ) -> EncoderOutput:
        """Encode batch of data into latent space."""
        # Get species-specific and global latent distributions
        encoded = self.encoder(
            batch,
            edge_index=self.homology_edges,
            edge_attr=self.homology_scores
        )
        return encoded

    def decode(
        self,
        encoded: EncoderOutput,
        batch: SparseExpressionData,
    ) -> torch.Tensor:
        """Decode latent vectors back to gene expression space."""
        return self.decoder(
            encoded, batch
        )

    def forward(self, batch: SparseExpressionData) -> torch.Tensor:
        """Forward pass through the model."""
        # Encode
        encoded = self.encode(batch)

        # Decode
        reconstruction = self.decode(encoded, batch)

        return reconstruction

    def homology_loss(
        self,
        encoded: EncoderOutput,
        batch: SparseExpressionData,
    ) -> torch.Tensor:
        device = encoded.global_latent.device

        if not hasattr(self, "homology_edges"):
            return torch.tensor(0.0, device=device)

        z = []
        for species_idx in range(self.n_species):
            if species_idx in encoded.species_latents:
                z.append(encoded.species_latents[species_idx].latent)
        z = torch.cat(z, dim=0)

        # Aggregate latent vectors per gene
        n_genes = self.n_genes
        with torch.no_grad():
            gene_latents = torch.zeros(n_genes, z.size(1), device=z.device)
            gene_counts = torch.zeros(n_genes, 1, device=z.device)
        gene_latents.index_add_(0, batch.gene_idx, z)
        gene_counts.index_add_(0, batch.gene_idx, torch.ones_like(z[:, :1]))
        gene_latents = gene_latents / gene_counts.clamp(min=1)

        # Normalize gene latents
        gene_latents = F.normalize(gene_latents, dim=1)

        # Get homology edge weights
        edge_weights = F.gumbel_softmax(
            self.homology_scores, tau=self.temperature, hard=True
        )

        # Compute positive similarities (homologous genes)
        pos_similarities = (
            F.cosine_similarity(
                gene_latents[self.homology_edges[:, 0]],
                gene_latents[self.homology_edges[:, 1]],
                dim=1,
            )
            / self.temperature
        )

        # Weighted positive loss
        pos_loss = -(
            edge_weights * torch.log(torch.sigmoid(pos_similarities) + 1e-8)
        ).mean()

        # Compute negative similarities with hard negative mining
        unique_genes = torch.unique(batch.gene_idx)
        if len(unique_genes) > 1:
            # Create mapping from original gene indices to new contiguous indices
            gene_to_idx = torch.zeros(n_genes, dtype=torch.long, device=z.device)
            gene_to_idx[unique_genes] = torch.arange(len(unique_genes), device=z.device)

            # Sample harder negatives by finding more similar non-homologous pairs
            with torch.no_grad():
                n_samples = min(len(unique_genes) * 10, 1000)
                unique_gene_latents = gene_latents[unique_genes]
                all_sims = torch.matmul(unique_gene_latents, unique_gene_latents.t())

                # Create mask for homology edges (using re-indexed edges)
                pos_mask = torch.zeros_like(all_sims, dtype=torch.bool)
                valid_edges_mask = torch.isin(
                    self.homology_edges[:, 0], unique_genes
                ) & torch.isin(self.homology_edges[:, 1], unique_genes)
                if valid_edges_mask.any():
                    valid_edges = self.homology_edges[valid_edges_mask]
                    remapped_edges = gene_to_idx[valid_edges]
                    pos_mask[remapped_edges[:, 0], remapped_edges[:, 1]] = True
                    pos_mask[remapped_edges[:, 1], remapped_edges[:, 0]] = (
                        True  # Symmetric
                    )

                all_sims.masked_fill_(pos_mask, float("-inf"))

                # Sample top-k similar negative pairs
                neg_sims, neg_indices = all_sims.view(-1).topk(n_samples)
                idx1 = neg_indices // len(unique_genes)
                idx2 = neg_indices % len(unique_genes)

            neg_similarities = (
                F.cosine_similarity(
                    unique_gene_latents[idx1], unique_gene_latents[idx2], dim=1
                )
                / self.temperature
            )

            neg_loss = -torch.log(1 - torch.sigmoid(neg_similarities) + 1e-8).mean()

            return (pos_loss + neg_loss) / 2

        return pos_loss

    def training_step(
        self, batch: SparseExpressionData, batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        # Add garbage collection if needed
        loss = self.compute_loss(batch)

        self.log("train_loss", loss.loss, sync_dist=True, batch_size=batch.batch_size)
        self.log("train_recon_loss", loss.recon_loss, sync_dist=True, batch_size=batch.batch_size)
        self.log("train_kl_loss", loss.kl_loss, sync_dist=True, batch_size=batch.batch_size)
        self.log("train_homology_loss", loss.homology_loss, sync_dist=True, batch_size=batch.batch_size)
        # Add memory logging at the end of training step
        self._log_memory_stats("train", batch.batch_size)

        # Add gradient norm logging
        if self.trainer.global_step % 100 == 0:  # Log every 100 steps
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"))
            self.log("gradient_norm", grad_norm, sync_dist=True)

        return loss.loss

    def compute_loss(self, batch: SparseExpressionData) -> LossOutput:
        import gc

        # Single forward pass through encoder
        encoded = self.encode(batch)
        reconstruction = self.decode(encoded, batch)

        # Reconstruction loss
        recon_loss = F.poisson_nll_loss(
            reconstruction, batch.values.view(-1, 1), reduction="mean"
        )

        # KL loss (using already computed distributions)
        kl_loss = 0.0
        for _, species_latent in encoded.species_latents.items():
            kl_loss += -0.5 * torch.mean(1 + species_latent.logvar - species_latent.mu.pow(2) - species_latent.logvar.exp())
        kl_loss += -0.5 * torch.mean(
            1 + encoded.global_logvar - encoded.global_mu.pow(2) - encoded.global_logvar.exp()
        )

        # Concatenate all latents for homology loss

        homology_loss = self.homology_loss(encoded, batch)

        # Total loss
        total_loss = recon_loss + kl_loss + homology_loss

        # Clear any cached tensors
        torch.cuda.empty_cache()
        gc.collect()        
        
        return LossOutput(
            loss=total_loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            homology_loss=homology_loss,
            reconstruction=reconstruction,
        )

    def validation_step(
        self, batch: SparseExpressionData, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        with torch.no_grad():
            loss = self.compute_loss(batch)

            # Additional validation metrics
            mse = F.mse_loss(loss.reconstruction, batch.values)
            mae = F.l1_loss(loss.reconstruction, batch.values)

        # Directly log metrics with sync_dist=True
        self.log("val_loss", loss.loss, sync_dist=True, batch_size=batch.batch_size)
        self.log("val_recon_loss", loss.recon_loss, sync_dist=True, batch_size=batch.batch_size)
        self.log("val_kl_loss", loss.kl_loss, sync_dist=True, batch_size=batch.batch_size)
        self.log("val_homology_loss", loss.homology_loss, sync_dist=True, batch_size=batch.batch_size)
        self.log("val_mse", mse, sync_dist=True, batch_size=batch.batch_size)
        self.log("val_mae", mae, sync_dist=True, batch_size=batch.batch_size)

        step_output = {
            "val_loss": loss.loss,
            "val_recon_loss": loss.recon_loss,
            "val_kl_loss": loss.kl_loss,
            "val_homology_loss": loss.homology_loss,
            "val_mse": mse,
            "val_mae": mae,
        }
        self.validation_step_outputs.append(step_output)

        # Add memory logging at the end of validation step
        self._log_memory_stats("val", batch.batch_size)

        return step_output

    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return

        # Stack all batch metrics
        metrics = {
            key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
            for key in self.validation_step_outputs[0].keys()
        }

        # Log metrics
        for key, value in metrics.items():
            self.log(key, value, sync_dist=True)

        # Clear stored outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer with improved learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Create custom learning rate schedule with warmup
        def lr_lambda(current_step: int):
            if self.warmup_steps is None:  # Default to 100 steps if not yet calculated
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
                    self.min_learning_rate / self.learning_rate,  # Convert to ratio
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
            total_samples = len(self.trainer.train_dataloader.dataset)
            steps_per_epoch = total_samples / self.global_batch_size
            self.warmup_steps = int(steps_per_epoch * self.warmup_epochs)
            print(f"Warmup steps calculated: {self.warmup_steps}")

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """Configure gradient clipping with improved settings."""
        # Use instance values if not provided
        gradient_clip_val = gradient_clip_val or self.gradient_clip_val
        gradient_clip_algorithm = (
            gradient_clip_algorithm or self.gradient_clip_algorithm
        )

        # Clip gradients
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def on_train_epoch_start(self):
        """Reset memory tracking at the start of each epoch."""
        self.prev_gpu_memory = 0
        if cuda.is_available():
            # Clear cache to get more accurate measurements
            cuda.empty_cache()
