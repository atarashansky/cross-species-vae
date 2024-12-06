from typing import Dict, Any, Union, List, Optional, Tuple
import anndata as ad
import pytorch_lightning as pl
import torch
from torch import cuda
import math
import torch.nn as nn
from torch.nn import functional as F
from src.dataclasses import BatchData
from src.data import CrossSpeciesInferenceDataset
from src.utils import negative_binomial_loss
from src.modules import Encoder, Decoder, FrozenEmbedding

class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    def __init__(
        self,
        species_vocab_sizes: Dict[int, int],
        homology_edges: Dict[int, Dict[int, torch.Tensor]] | None = None,
        homology_scores: Dict[int, Dict[int, torch.Tensor]] | None = None,
        embedding_weights: Dict[int, torch.Tensor] | None = None,
        n_latent: int = 256,
        hidden_dims: list = [256],
        dropout_rate: float = 0.2,
        base_learning_rate: float = 8e-3,
        base_batch_size: int = 256,
        batch_size: int = 256,
        min_learning_rate: float = 1e-4,
        warmup_epochs: float = 1.0,
        init_beta: float = 1e-3,
        final_beta: float = 1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        direct_recon_weight: float = 1.0,
        cross_species_recon_weight: float = 1.0,
        homology_weight: float = 0.0,
        cycle_weight: float = 0.0,
        transform_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['homology_edges', 'homology_scores', 'embedding_weights'])
        
        # Scale learning rate based on batch size
        self.learning_rate = base_learning_rate
        self.direct_recon_weight = direct_recon_weight
        self.cross_species_recon_weight = cross_species_recon_weight
        self.homology_weight = homology_weight
        self.cycle_weight = cycle_weight
        self.transform_weight = transform_weight
        self.init_beta = init_beta
        self.final_beta = final_beta
        self.current_beta = init_beta  # Track current beta value
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        
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
                embedding=FrozenEmbedding(embedding_weights[species_id]) if embedding_weights is not None else None,
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
            self.homology_edges = nn.ModuleDict({
                str(src_id): nn.ModuleDict({
                    str(dst_id): self.register_buffer(
                        f'edges_{src_id}_{dst_id}',
                        edges if isinstance(edges, torch.Tensor) else torch.tensor(edges)
                    )
                    for dst_id, edges in species_edges.items()
                })
                for src_id, species_edges in homology_edges.items()
            })
            
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
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"].detach(), sync_dist=True)
        self.log("train_direct_recon", loss_dict["direct_recon"].detach(), sync_dist=True)
        self.log("train_cross_species_recon", loss_dict["cross_species_recon"].detach(), sync_dist=True)
        self.log("train_kl", loss_dict["kl"].detach(), sync_dist=True)
        self.log("train_homology", loss_dict["homology"].detach(), sync_dist=True)
        self.log("train_cycle", loss_dict["cycle"].detach(), sync_dist=True)
        self.log("train_transform", loss_dict["transform"].detach(), sync_dist=True)

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
            "val_homology": loss_dict["homology"].detach(),
            "val_cycle": loss_dict["cycle"].detach(),
            "val_transform": loss_dict["transform"].detach(),
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
        """Reset memory tracking at the start of each epoch."""        
        self.prev_gpu_memory = 0
        if cuda.is_available():
            cuda.empty_cache()

        
    def _compute_transform_consistency_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData
    ) -> torch.Tensor:
        transform_loss = torch.tensor(0.0, device=self.device)

        if self.transform_weight == 0.0:
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
            
    def _compute_cycle_consistency_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData
    ) -> torch.Tensor:
        """Compute cycle consistency loss between encoders.
        
        For each source species:
            1. Get original encoding from source encoder
            2. Pass through each target decoder
            3. Re-encode with target encoder
            4. Compare original and re-encoded representations
        """
        cycle_loss = torch.tensor(0.0, device=self.device)

        if not hasattr(self, 'homology_scores') or self.cycle_weight == 0.0:
            return cycle_loss
        
        counter = 0
        
        for src_species_id in batch.data:
            # Get original encoding
            original_encoding = outputs[src_species_id]['encoder_outputs'][src_species_id]
            homology_reconstructions = outputs[src_species_id]['homology_reconstructions']
            
            # For each target species (except source)
            for dst_species_id in self.species_vocab_sizes.keys():
                if dst_species_id == src_species_id:
                    continue
                    
                # Pass through target decoder
                decoded = homology_reconstructions[dst_species_id]
                
                # Re-encode with target encoder
                dst_encoder = self.encoders[str(dst_species_id)]
                reencoded = dst_encoder(decoded['mean'])
                
                cycle_loss += torch.mean(
                    0.5 * (
                        torch.exp(reencoded['logvar'] - original_encoding['logvar'])
                        + (original_encoding['mu'] - reencoded['mu']).pow(2) / torch.exp(original_encoding['logvar'])
                        - 1
                        + original_encoding['logvar']
                        - reencoded['logvar']
                    )
                )

                counter += 1
        
        return self.cycle_weight * cycle_loss / counter   
    
    def _compute_homology_loss(self, outputs: Dict[int, Any], batch: BatchData) -> torch.Tensor:   
        homology_loss = torch.tensor(0.0, device=self.device)

        if not hasattr(self, 'homology_scores') or self.homology_weight == 0.0:
            return homology_loss
        
        counter = 0
        # Compute homology loss across species pairs
        for src_species_id in batch.data:
            for dst_species_id in batch.data:
                if src_species_id == dst_species_id:
                    continue
                
                # Get edges and scores for this species pair
                edges = getattr(self, f'edges_{src_species_id}_{dst_species_id}')
                scores = torch.sigmoid(self.homology_scores[str(src_species_id)][str(dst_species_id)].detach())

                src, dst = edges.t()
                                
                src_data = outputs[src_species_id]['homology_reconstructions'][src_species_id]['mean']
                dst_data = outputs[src_species_id]['homology_reconstructions'][dst_species_id]['mean']
                
                src_pred = src_data[:, src]
                dst_pred = dst_data[:, dst]

                # Center the predictions
                src_centered = src_pred - src_pred.mean(dim=0)
                dst_centered = dst_pred - dst_pred.mean(dim=0)

                # Compute correlation
                covariance = (src_centered * dst_centered).mean(dim=0)
                src_std = src_centered.std(dim=0)
                dst_std = dst_centered.std(dim=0)
                correlation = covariance / (src_std * dst_std + 1e-8)

                alignment_loss = torch.mean(
                    scores * (1 - correlation.clamp(min=0)) 
                    # + (1 - scores) * correlation.clamp(min=0) # TODO uncomment
                )
                
                homology_loss += alignment_loss
                counter += 1

        return self.homology_weight * homology_loss / counter


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

        if self.cross_species_recon_weight == 0.0:
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
    


    def compute_loss(self, batch: BatchData, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        direct_recon_loss = self._compute_direct_reconstruction_loss(outputs, batch)
        cross_species_recon_loss = self._compute_cross_species_reconstruction_loss(outputs, batch)
        
        kl = self._compute_kl_loss(outputs)

        homology_loss = self._compute_homology_loss(outputs, batch)

        cycle_loss = self._compute_cycle_consistency_loss(outputs, batch)

        transform_loss = self._compute_transform_consistency_loss(outputs, batch)
        

        beta = self.get_current_beta()
        total_loss = direct_recon_loss + cross_species_recon_loss + homology_loss + cycle_loss + transform_loss + beta * kl
        
        return {
            "loss": total_loss,
            "direct_recon": direct_recon_loss,
            "cross_species_recon": cross_species_recon_loss,
            "kl": kl,
            "homology": homology_loss,
            "cycle": cycle_loss,
            "transform": transform_loss,
        }
    
    
    def _transform_expression(
        self,
        batch: BatchData,
        src_species: int,
        dst_species: int,
    ) -> torch.Tensor:
        # Create dense transformation matrix
        edges = getattr(self, f'edges_{src_species}_{dst_species}')
        scores = torch.sigmoid(self.homology_scores[str(src_species)][str(dst_species)])
        
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
        return transformed


   
    def forward(self, batch: BatchData) -> Dict[str, Any]:
        results = {}
        
        for source_species_id, source_data in batch.data.items():
            # For each source species (A, B, C)            
            reconstructions = {}
            encoder_outputs = {}
                        
            source_encoder = self.encoders[str(source_species_id)]
            source_embedded = source_encoder.embed(source_data)
            source_outputs = source_encoder.encode(source_embedded)
            encoder_outputs[source_species_id] = source_outputs

            
            for target_species_id in self.species_vocab_sizes.keys():
                if target_species_id == source_species_id:
                    continue
                                        
                # Encode transformed data with target encoder
                target_encoder = self.encoders[str(target_species_id)]
                target_encoded = target_encoder.encode(source_embedded)
                encoder_outputs[target_species_id] = target_encoded
                 
            
            source_decoder = self.decoders[str(source_species_id)]
            for target_species_id in self.species_vocab_sizes.keys():
                reconstructions[target_species_id] = source_decoder(encoder_outputs[target_species_id]['z'])
            

            results[source_species_id] = {
                'encoder_outputs': encoder_outputs,
                'reconstructions': reconstructions,
            }
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


