from typing import Dict, Any, Union, List, Optional, Tuple
import anndata as ad
import pytorch_lightning as pl
import torch
from torch import cuda
import numpy as np
import math
import hnswlib
import torch.nn as nn
from torch.nn import functional as F
from src.dataclasses import BatchData
from src.data import CrossSpeciesInferenceDataset
from src.utils import negative_binomial_loss
from src.modules import Encoder, Decoder

class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    def __init__(
        self,
        species_vocab_sizes: Dict[int, int],
        homology_edges: Dict[int, Dict[int, torch.Tensor]] | None = None,
        homology_scores: Dict[int, Dict[int, torch.Tensor]] | None = None,
        n_latent: int = 256,
        hidden_dims: list = [256],
        dropout_rate: float = 0.2,
        base_learning_rate: float = 1e-3,
        base_batch_size: int = 256,
        batch_size: int = 256,
        min_learning_rate: float = 1e-5,
        warmup_data: float = 0.1,
        init_beta: float = 1e-3,
        final_beta: float = 1,
        triplet_epoch_start: float = 0.5,
        triplet_loss_margin: float = 0.2,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        direct_recon_weight: float = 1.0,
        cross_species_recon_weight: float = 1.0,
        transform_weight: float = 0.1,
        homology_score_momentum: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['homology_edges', 'homology_scores'])
        
        # Scale learning rate based on batch size
        self.learning_rate = base_learning_rate
        self.direct_recon_weight = direct_recon_weight
        self.cross_species_recon_weight = cross_species_recon_weight
        self.transform_weight = transform_weight
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.current_beta = init_beta  # Track current beta value
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        self.triplet_loss_margin = triplet_loss_margin
        self.triplet_epoch_start = triplet_epoch_start
        self.homology_score_momentum = homology_score_momentum
        
        # Store parameters
        self.species_vocab_sizes = species_vocab_sizes
        self.n_species = len(species_vocab_sizes)
        self.n_latent = n_latent

        self.mu_layer = nn.Linear(hidden_dims[-1], n_latent)
        self.logvar_layer = nn.Linear(hidden_dims[-1], n_latent)
                
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
                hidden_dims=hidden_dims[::-1],  # Reverse hidden dims for decoder
                dropout_rate=dropout_rate,
            )
            for species_id, vocab_size in species_vocab_sizes.items()
        })

        # Register homology information
        self.homology_available = homology_edges is not None
        if self.homology_available:
            # Register edges as buffers
            for src_id, species_edges in homology_edges.items():
                for dst_id, edges in species_edges.items():
                    self.register_buffer(
                        f'edges_{src_id}_{dst_id}',
                        edges if isinstance(edges, torch.Tensor) else torch.tensor(edges)
                    )
            
            # Initialize and register score buffers
            for src_id, species_edges in homology_edges.items():
                for dst_id, edges in species_edges.items():                    
                    init_scores = (
                        torch.ones(len(edges), dtype=torch.float32) if homology_scores is None
                        else homology_scores[src_id][dst_id] / homology_scores[src_id][dst_id].max()
                    )
                    self.register_buffer(f'scores_{src_id}_{dst_id}', init_scores)
                    
                    # Initialize running statistics for each species
                    self.register_buffer(
                        f'running_sum_products_{src_id}_{dst_id}',
                        torch.zeros(len(edges), dtype=torch.float32)
                    )
                    self.register_buffer(
                        f'running_sums_src_{src_id}',
                        torch.zeros(species_vocab_sizes[src_id], dtype=torch.float32)
                    )
                    self.register_buffer(
                        f'running_sums_dst_{dst_id}',
                        torch.zeros(species_vocab_sizes[dst_id], dtype=torch.float32)
                    )
                    self.register_buffer(
                        f'running_sum_squares_src_{src_id}',
                        torch.zeros(species_vocab_sizes[src_id], dtype=torch.float32)
                    )
                    self.register_buffer(
                        f'running_sum_squares_dst_{dst_id}',
                        torch.zeros(species_vocab_sizes[dst_id], dtype=torch.float32)
                    )
            
            self.register_buffer('running_count', torch.zeros(1))

        
        # Training parameters
        self.min_learning_rate = min_learning_rate
        self.warmup_data = warmup_data

        self.warmup_steps = None  # Will be set in on_train_start
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.validation_step_outputs = []

        # Initialize species-pair specific running counts
        for src_id in species_vocab_sizes.keys():
            for dst_id in species_vocab_sizes.keys():
                if src_id == dst_id:
                    continue
                self.register_buffer(
                    f'running_count_{src_id}_{dst_id}',
                    torch.zeros(1)
                )


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
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay with minimum learning rate
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                )
                return max(
                    self.min_learning_rate / self.learning_rate,
                    0.5 * (1.0 + math.cos(math.pi * progress * 0.85)), # TODO: slower cosine decay
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
    

    def training_step(self, batch: BatchData, batch_idx: int):     
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        if batch.triplets is None and batch.data is not None:
            self._update_correlation_estimates(batch, outputs)
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"].detach(), sync_dist=True)
        self.log("train_direct_recon", loss_dict["direct_recon"].detach(), sync_dist=True)
        self.log("train_cross_species_recon", loss_dict["cross_species_recon"].detach(), sync_dist=True)
        self.log("train_kl", loss_dict["kl"].detach(), sync_dist=True)
        self.log("train_transform", loss_dict["transform"].detach(), sync_dist=True)
        self.log("train_triplet", loss_dict["triplet"].detach(), sync_dist=True)

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
            "val_direct_recon": loss_dict["direct_recon"].detach(),
            "val_cross_species_recon": loss_dict["cross_species_recon"].detach(),
            "val_kl": loss_dict["kl"].detach(),
            "val_transform": loss_dict["transform"].detach(),
            "val_triplet": loss_dict["triplet"].detach(),
        })
        
                        
        return loss_dict["loss"]

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

    def on_train_epoch_start(self):
        """Reset correlation tracking at the start of each epoch."""        
        self.prev_gpu_memory = 0
        if cuda.is_available():
            cuda.empty_cache()
     
        # Reset running statistics
        for src_id in self.species_vocab_sizes.keys():
            for dst_id in self.species_vocab_sizes.keys():
                if src_id == dst_id:
                    continue
                    
                getattr(self, f'running_sum_products_{src_id}_{dst_id}').zero_()
                
                getattr(self, f'running_sums_src_{src_id}').zero_()
                getattr(self, f'running_sums_dst_{dst_id}').zero_()
                
                getattr(self, f'running_sum_squares_src_{src_id}').zero_()
                getattr(self, f'running_sum_squares_dst_{dst_id}').zero_()
                
                getattr(self, f'running_count_{src_id}_{dst_id}').zero_()
                getattr(self, f'running_count_{dst_id}_{src_id}').zero_()
            

    def on_train_epoch_end(self):
        """Update homology scores based on accumulated correlations"""
        with torch.no_grad():
            keys = list(self.species_vocab_sizes.keys())
            for i, src_species_id in enumerate(keys):
                for j, dst_species_id in enumerate(keys):
                    if j <= i:
                        continue
                    
                    running_count = getattr(self, f'running_count_{src_species_id}_{dst_species_id}')
                    # Get edges for this species pair
                    edges = getattr(self, f'edges_{src_species_id}_{dst_species_id}')
                    
                    # Compute means with epsilon for stability
                    eps = 1e-8
                    src_means = getattr(self, f'running_sums_src_{src_species_id}')[edges[:, 0]] / (running_count + eps)
                    dst_means = getattr(self, f'running_sums_dst_{dst_species_id}')[edges[:, 1]] / (running_count + eps)
                    
                    # Get sum of products and compute mean
                    sum_products = getattr(self, f'running_sum_products_{src_species_id}_{dst_species_id}')
                    mean_products = sum_products / (running_count + eps)
                    covariance = mean_products - (src_means * dst_means)                    
                    
                    # Compute variances using sum of squares
                    src_mean_squares = getattr(self, f'running_sum_squares_src_{src_species_id}')[edges[:, 0]] / (running_count + eps)
                    dst_mean_squares = getattr(self, f'running_sum_squares_dst_{dst_species_id}')[edges[:, 1]] / (running_count + eps)
                    
                    # Add epsilon to variances for numerical stability
                    src_var = (src_mean_squares - src_means.pow(2)).clamp(min=eps)
                    dst_var = (dst_mean_squares - dst_means.pow(2)).clamp(min=eps)
                    
                    # Compute correlation with stable denominator                    
                    correlations = covariance / torch.sqrt(src_var * dst_var + eps)
                    new_scores = correlations.clamp(0, 1)  # Ensure valid range
                    
                    # Get current scores
                    scores = getattr(self, f'scores_{src_species_id}_{dst_species_id}')

                    # Apply momentum update
                    scores.data = self.homology_score_momentum * scores + (1 - self.homology_score_momentum) * new_scores

                    # Update symmetric scores
                    scores_symmetric = getattr(self, f'scores_{dst_species_id}_{src_species_id}')
                    scores_symmetric.data = scores.data
        
        if self.trainer.current_epoch >= self.triplet_epoch_start * self.trainer.max_epochs :
            self.get_triplets_for_current_epoch()

        if cuda.is_available():
            cuda.empty_cache()



    def get_train_dataset(self):
        """Get the training dataset from the trainer's datamodule."""
        if self.trainer is None or self.trainer.datamodule is None:
            return None
        return self.trainer.datamodule.train_dataset

    @torch.no_grad()
    def get_triplets_for_current_epoch(self):
        """Get triplets for current epoch using latent embeddings."""
        dataset = self.get_train_dataset()
        dataset.triplets = None

        latents = {k: [] for k in self.species_vocab_sizes.keys()}
        labels = {k: [] for k in self.species_vocab_sizes.keys()}

        for batch in dataset:
            for target_species_id in batch.data:
                # Move data to same device as model
                data = batch.data[target_species_id].to(self.device)
                mu = self.encoders[str(target_species_id)](data)['mu'].cpu()
                latents[target_species_id].append(mu)
                labels[target_species_id].append(batch.labels[target_species_id])
        
        for target_species_id in latents:
            latents[target_species_id] = torch.cat(latents[target_species_id], dim=0)
            labels[target_species_id] = np.concatenate(labels[target_species_id])
        
        triplets = {}
        
        num_triplets = 0
        for src_species_id in self.species_vocab_sizes.keys():
            for dst_species_id in self.species_vocab_sizes.keys():
                if src_species_id == dst_species_id:
                    continue
                
                valid_anchors, valid_positives, valid_negatives = self._mine_triplets_one_nn(
                    latents, labels, src_species_id, dst_species_id
                )
                
                triplets[(src_species_id, dst_species_id)] = {
                    'anchor': valid_anchors,
                    'positive': valid_positives,
                    'negative': valid_negatives
                }

                num_triplets += len(valid_anchors)

        print(f"Found {num_triplets} triplets")
        dataset.triplets = triplets

    def _mine_triplets_one_nn(self, latents, labels, src_id, dst_id, ef=200, M=48):
        emb1 = latents[src_id].numpy()
        emb2 = latents[dst_id].numpy()

        # Get nearest neighbors in emb2 for each cell in emb1
        labels2 = np.arange(emb2.shape[0])
        p2 = hnswlib.Index(space='cosine', dim=emb2.shape[1])
        p2.init_index(max_elements=emb2.shape[0], ef_construction=ef, M=M)
        p2.add_items(emb2, labels2)
        p2.set_ef(ef)
        idx1, _ = p2.knn_query(emb1, k=1)
        idx1 = idx1.squeeze()
        # Get nearest neighbors in emb1 for each neighbor of each cell in emb1
        labels1 = np.arange(emb1.shape[0])
        p1 = hnswlib.Index(space='cosine', dim=emb1.shape[1])
        p1.init_index(max_elements=emb1.shape[0], ef_construction=ef, M=M)
        p1.add_items(emb1, labels1)
        p1.set_ef(ef)
        idx2, _ = p1.knn_query(emb2[idx1], k=1)   
        idx2 = idx2.squeeze()

        # Find positive pairs as labels1[idx2] == labels1
        valid_pairs = labels[src_id][idx2] == labels[src_id]
        anchors = np.where(valid_pairs)[0]
        positives = idx1[anchors]

        valid_negatives = []
        valid_positives = []
        valid_anchors = []
        for anchor_idx, positive_idx in zip(anchors, positives):
            anchor_label = labels[src_id][anchor_idx]
            
            # Find cells in src_species (same as anchor) with different labels
            neg_candidates = np.where(labels[src_id] != anchor_label)[0]
            
            if len(neg_candidates) > 0:
                # Randomly select one negative example
                neg_idx = np.random.choice(neg_candidates)
                valid_negatives.append(neg_idx)   
                valid_positives.append(positive_idx)
                valid_anchors.append(anchor_idx)
        
        # Anchors: src
        # Positives: dst
        # Negatives: src
        return np.array(valid_anchors), np.array(valid_positives), np.array(valid_negatives)


    def _compute_transform_consistency_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData
    ) -> torch.Tensor:
        transform_loss = torch.tensor(0.0, device=self.device)

        if not self.homology_available or self.transform_weight == 0.0:
            return transform_loss
        
        counter = 0
        
        for src_species_id in batch.data:
            # Get original encoding
            original_encoding = outputs[src_species_id]['encoder_outputs'][src_species_id]
            
            # For each target species (except source)
            for dst_species_id in self.species_vocab_sizes.keys():
                if dst_species_id == src_species_id:
                    continue

                new_encoding = outputs[src_species_id]['encoder_outputs'][dst_species_id]
                
                transform_loss += torch.mean(
                    0.5 * (
                        torch.exp(new_encoding['logvar'] - original_encoding['logvar'])
                        + (original_encoding['mu'] - new_encoding['mu']).pow(2) / torch.exp(original_encoding['logvar'])
                        - 1
                        + original_encoding['logvar']
                        - new_encoding['logvar']
                    )
                )

                counter += 1
        
        return self.transform_weight * transform_loss / counter 
  
    def _compute_direct_reconstruction_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData,
    ) -> torch.Tensor:
    
        count_loss_nb = torch.tensor(0.0, device=self.device)

        if self.direct_recon_weight == 0.0:
            return count_loss_nb
        
        counter = 0
        
        for target_species_id in outputs:
            reconstructions = outputs[target_species_id]['reconstructions']
            
            target_counts = batch.data[target_species_id]
            target_counts = torch.exp(target_counts) - 1
            target_norm = target_counts / target_counts.sum(dim=1, keepdim=True) * 10_000                  

            pred_counts = torch.exp(reconstructions[target_species_id]['mean']) - 1
            pred_norm = pred_counts / pred_counts.sum(dim=1, keepdim=True) * 10_000            
            
            count_loss_nb += negative_binomial_loss(
                pred=pred_norm.clamp(min=1e-6),
                target=target_norm,
                theta=reconstructions[target_species_id]['theta'],
            )
            counter += 1
            
        return self.direct_recon_weight * count_loss_nb / counter
    
    def _compute_cross_species_reconstruction_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData,
    ) -> torch.Tensor:
    
        count_loss_nb = torch.tensor(0.0, device=self.device)

        if not self.homology_available or self.cross_species_recon_weight == 0.0:
            return count_loss_nb
        
        counter = 0
        
        for target_species_id in outputs:
            reconstructions = outputs[target_species_id]['reconstructions']
            
            target_counts = batch.data[target_species_id]
            target_counts = torch.exp(target_counts) - 1
            target_norm = target_counts / target_counts.sum(dim=1, keepdim=True) * 10_000                  

            for dst_species_id, recon in reconstructions.items():
                if dst_species_id == target_species_id:
                    continue
                
                pred_counts = torch.exp(recon['mean']) - 1
                pred_norm = pred_counts / pred_counts.sum(dim=1, keepdim=True) * 10_000            
                
                count_loss_nb += negative_binomial_loss(
                    pred=pred_norm.clamp(min=1e-6),
                    target=target_norm,
                    theta=recon['theta'],
                )
                counter += 1
    
        return self.cross_species_recon_weight * count_loss_nb / counter
    
    def _compute_kl_loss(
        self,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        kl = torch.tensor(0.0, device=self.device)
        counter = 0
        for target_species_id in outputs:
            # Get encoder outputs for all species
            encoder_outputs = outputs[target_species_id]['encoder_outputs']
            
            # Compute KL for each encoding
            for _, enc_output in encoder_outputs.items():
                mu = enc_output['mu']
                logvar = enc_output['logvar']
                
                # Add numerical stability clamps
                logvar = torch.clamp(logvar, min=-20, max=20)  # Prevent extreme values
                mu = torch.clamp(mu, min=-20, max=20)  # Prevent extreme values
                
                kl += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                counter += 1
                
        return kl / counter
    
    def _compute_triplet_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """Compute triplet loss with cosine distance.
        
        Loss = max(D(z_a, z_p) - D(z_a, z_n) + margin, 0)
        where D is cosine distance: D(x,y) = 1 - cos(x,y)
        """
        margin = self.triplet_loss_margin
        
        # Get latent vectors
        z_anchor = outputs["anchor_encoded"]["z"]
        z_positive = outputs["positive_encoded"]["z"]
        z_negative = outputs["negative_encoded"]["z"]
        
        # Compute cosine distances
        # Note: cosine_similarity returns values in [-1, 1], so we need 1 - cos to get distance
        pos_dist = 1 - F.cosine_similarity(z_anchor, z_positive)
        neg_dist = 1 - F.cosine_similarity(z_anchor, z_negative)
        
        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0).mean()
        
        # Return mean loss over batch
        return {
            "loss": loss,
            "direct_recon": torch.tensor(0.0, device=self.device),
            "cross_species_recon": torch.tensor(0.0, device=self.device),
            "kl": torch.tensor(0.0, device=self.device),
            "transform": torch.tensor(0.0, device=self.device),   
            "triplet": loss,             
        }
    
    def compute_loss(self, batch: BatchData, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if batch.triplets is not None:
            return self._compute_triplet_loss(outputs)
        
        direct_recon_loss = self._compute_direct_reconstruction_loss(outputs, batch)

        cross_species_recon_loss = self._compute_cross_species_reconstruction_loss(outputs, batch)
                
        kl = self._compute_kl_loss(outputs)

        transform_loss = self._compute_transform_consistency_loss(outputs, batch)
                
        beta = self.get_current_beta()
        total_loss = direct_recon_loss + cross_species_recon_loss + transform_loss + beta * kl
        
        return {
            "loss": total_loss,
            "direct_recon": direct_recon_loss,
            "cross_species_recon": cross_species_recon_loss,
            "kl": kl,
            "transform": transform_loss,
            "triplet": torch.tensor(0.0, device=self.device),
        }
    
    @torch.no_grad()
    def _transform_expression(
        self,
        batch: BatchData,
        src_species: int,
        dst_species: int,
    ) -> torch.Tensor:
        # Create dense transformation matrix
        edges = getattr(self, f'edges_{src_species}_{dst_species}')
        scores = getattr(self, f'scores_{src_species}_{dst_species}') 
        
        n_src_genes = self.species_vocab_sizes[src_species]
        n_dst_genes = self.species_vocab_sizes[dst_species]
        transform_matrix = torch.zeros(
            n_dst_genes,
            n_src_genes,
            device=batch.data[src_species].device
        )
        transform_matrix[edges[:, 1], edges[:, 0]] = scores
        # Normalize transform matrix so src edges sum to 1.0 per dst gene
        transform_matrix = transform_matrix / (transform_matrix.sum(dim=1, keepdim=True).clamp(min=1e-3))

        x = batch.data[src_species]

        # Transform expression data
        transformed = x @ transform_matrix.t()
        
        # Explicitly delete the transform matrix
        del transform_matrix
        torch.cuda.empty_cache()  # Force CUDA to release memory
        
        return transformed


   
    def forward(self, batch: BatchData) -> Dict[str, Any]:
        results = {}        
        if batch.data is not None:
            for source_species_id, source_data in batch.data.items():
                # For each source species (A, B, C)            
                reconstructions = {}
                homology_reconstructions = {}
                encoder_outputs = {}
                            
                source_encoder = self.encoders[str(source_species_id)]
                source_encoded = source_encoder(source_data)
                encoder_outputs[source_species_id] = source_encoded
                
                if self.homology_available:
                    for target_species_id in self.species_vocab_sizes.keys():
                        if target_species_id == source_species_id:
                            continue
                            
                        # Transform source data to target space
                        transformed = self._transform_expression(batch, source_species_id, target_species_id)
                        
                        # Encode transformed data with target encoder
                        target_encoder = self.encoders[str(target_species_id)]
                        transformed_encoded = target_encoder(transformed)
                        encoder_outputs[target_species_id] = transformed_encoded
                                                    
                
                source_decoder = self.decoders[str(source_species_id)]
                # Now decode using concatenated latents for each target space
                for target_species_id in self.species_vocab_sizes.keys():
                    # Decode with target decoder using all latents
                    reconstructions[target_species_id] = source_decoder(encoder_outputs[target_species_id]['z'])
                

                for target_species_id in self.species_vocab_sizes.keys():
                    homology_reconstructions[target_species_id] = self.decoders[str(target_species_id)](encoder_outputs[source_species_id]['mu'])

                results[source_species_id] = {
                    'encoder_outputs': encoder_outputs,
                    'reconstructions': reconstructions,
                    'homology_reconstructions': homology_reconstructions,
                }
        elif batch.triplets is not None:
            for (src_species_id, dst_species_id), triplet_data in batch.triplets.items():
                anchor_data = triplet_data['anchor']
                positive_data = triplet_data['positive']
                negative_data = triplet_data['negative']

                anchor_encoded = self.encoders[str(src_species_id)](anchor_data)
                positive_encoded = self.encoders[str(dst_species_id)](positive_data)
                negative_encoded = self.encoders[str(src_species_id)](negative_data)

                results["anchor_encoded"] = anchor_encoded
                results["positive_encoded"] = positive_encoded
                results["negative_encoded"] = negative_encoded
            
            results["src_species_id"] = src_species_id
            results["dst_species_id"] = dst_species_id

        return results

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
                data={k: v.to(device) for k,v in batch.data.items()},
            )
            for target_species_id in batch.data:
                latents = self.encoders[str(target_species_id)](batch.data[target_species_id])['z'].cpu()
                all_latents.append(latents)
                
                if return_species:
                    species_idx = torch.full((latents.shape[0],), target_species_id, dtype=torch.long, device=device)
                    all_species.append(species_idx)
        
        # Concatenate all batches
        latents = torch.cat(all_latents, dim=0)

        if return_species:
            species = torch.cat(all_species, dim=0)
            return latents, species
        
        return latents

    def _update_correlation_estimates(self, batch: BatchData, outputs: Dict[str, Any]):
        """Update running sums for gene-gene correlations across species"""
        
        keys = list(batch.data.keys())
        for i, src_species_id in enumerate(keys):
            src_data = batch.data[src_species_id]
            # Update source gene sums
            running_sums_src = getattr(self, f'running_sums_src_{src_species_id}')
            running_sums_src.add_(src_data.sum(dim=0).detach())
            
            # For each target species
            for j, dst_species_id in enumerate(keys):
                if j <= i:
                    continue
                    
                # Get reconstructed expression in target space
                dst_data = outputs[src_species_id]['homology_reconstructions'][dst_species_id]['mean'].detach()
                
                # Update target gene sums
                running_sums_dst = getattr(self, f'running_sums_dst_{dst_species_id}')
                running_sums_dst.add_(dst_data.sum(dim=0))
                
                # Get edges for this species pair
                edges = getattr(self, f'edges_{src_species_id}_{dst_species_id}')
                
                # Update raw sum of products - compute intermediate result first
                products = (src_data[:, edges[:, 0]] * dst_data[:, edges[:, 1]]).sum(dim=0).detach()
                running_sum_products = getattr(self, f'running_sum_products_{src_species_id}_{dst_species_id}')
                running_sum_products.add_(products)

                # Add tracking of squared values - compute intermediate results first
                src_squares = src_data.pow(2).sum(dim=0).detach()
                dst_squares = dst_data.pow(2).sum(dim=0).detach()
                
                running_sum_squares_src = getattr(self, f'running_sum_squares_src_{src_species_id}')
                running_sum_squares_src.add_(src_squares)

                running_sum_squares_dst = getattr(self, f'running_sum_squares_dst_{dst_species_id}')
                running_sum_squares_dst.add_(dst_squares)

                # Update count for this specific species pair
                running_count = getattr(self, f'running_count_{src_species_id}_{dst_species_id}')
                running_count += batch.data[src_species_id].shape[0]