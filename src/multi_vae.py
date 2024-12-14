from typing import Dict, Any, Union, List, Optional, Tuple
import anndata as ad
import torch
from torch import cuda
import torch.nn as nn
from torch.nn import functional as F
from src.dataclasses import BatchData
from src.data import CrossSpeciesInferenceDataset
from src.utils import negative_binomial_loss
from src.modules import Encoder, Decoder
from src.base_model import BaseModel


class CrossSpeciesVAE(BaseModel):
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
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        direct_recon_weight: float = 1.0,
        cross_species_recon_weight: float = 1.0,
        transform_weight: float = 0.1,
        homology_score_momentum: float = 0.9,
    ):
        super().__init__(
            base_learning_rate=base_learning_rate,
            base_batch_size=base_batch_size,
            batch_size=batch_size,
            min_learning_rate=min_learning_rate,
            warmup_data=warmup_data,
            init_beta=init_beta,
            final_beta=final_beta,                 
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,               
        )

        self.save_hyperparameters(ignore=['homology_edges', 'homology_scores'])
        
        
        # Store parameters
        self.direct_recon_weight = direct_recon_weight
        self.cross_species_recon_weight = cross_species_recon_weight
        self.transform_weight = transform_weight
        self.homology_score_momentum = homology_score_momentum        
        self.species_vocab_sizes = species_vocab_sizes
        self.n_species = len(species_vocab_sizes)
        self.n_latent = n_latent
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        # Initialize model components
        self._init_model_components()
        
        self.homology_available = homology_edges is not None
        if self.homology_available:
            self._register_homology_buffers(homology_edges, homology_scores)

    def _register_homology_buffers(self, homology_edges, homology_scores):        
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
                    torch.zeros(self.species_vocab_sizes[src_id], dtype=torch.float32)
                )
                self.register_buffer(
                    f'running_sums_dst_{dst_id}',
                    torch.zeros(self.species_vocab_sizes[dst_id], dtype=torch.float32)
                )
                self.register_buffer(
                    f'running_sum_squares_src_{src_id}',
                    torch.zeros(self.species_vocab_sizes[src_id], dtype=torch.float32)
                )
                self.register_buffer(
                    f'running_sum_squares_dst_{dst_id}',
                    torch.zeros(self.species_vocab_sizes[dst_id], dtype=torch.float32)
                )
        
        self.register_buffer('running_count', torch.zeros(1))

        # Initialize species-pair specific running counts
        for src_id in self.species_vocab_sizes.keys():
            for dst_id in self.species_vocab_sizes.keys():
                if src_id == dst_id:
                    continue
                self.register_buffer(
                    f'running_count_{src_id}_{dst_id}',
                    torch.zeros(1)
                )

    def _init_model_components(self):
        self.mu_layer = nn.Linear(self.hidden_dims[-1], self.n_latent)
        self.logvar_layer = nn.Linear(self.hidden_dims[-1], self.n_latent)

        self.encoders = nn.ModuleDict({
            str(species_id): Encoder(
                n_genes=vocab_size,
                mu_layer=self.mu_layer,
                logvar_layer=self.logvar_layer,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate,
            )
            for species_id, vocab_size in self.species_vocab_sizes.items()
        })
    
        self.decoders = nn.ModuleDict({
            str(species_id): Decoder(
                n_genes=vocab_size,
                n_latent=self.n_latent,
                hidden_dims=self.hidden_dims[::-1],  # Reverse hidden dims for decoder
                dropout_rate=self.dropout_rate,
            )
            for species_id, vocab_size in self.species_vocab_sizes.items()
        })    

    def training_step(self, batch: BatchData, batch_idx: int):     
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        self._update_correlation_estimates(batch, outputs)
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"].detach(), sync_dist=True)
        self.log("train_direct_recon", loss_dict["direct_recon"].detach(), sync_dist=True)
        self.log("train_cross_species_recon", loss_dict["cross_species_recon"].detach(), sync_dist=True)
        self.log("train_kl", loss_dict["kl"].detach(), sync_dist=True)
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
            "val_transform": loss_dict["transform"].detach(),
        })
        
        return loss_dict["loss"]


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

        if cuda.is_available():
            cuda.empty_cache()



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
                        torch.exp(original_encoding['logvar'] - new_encoding['logvar'])  # σ₁²/σ₂²
                        + (original_encoding['mu'] - new_encoding['mu']).pow(2) / torch.exp(new_encoding['logvar'])  # (μ₁-μ₂)²/σ₂²
                        - 1  # constant term
                        + new_encoding['logvar'] - original_encoding['logvar']  # ln(σ₂²/σ₁²)
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
    

    def compute_loss(self, batch: BatchData, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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