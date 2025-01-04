from typing import Dict, Any, Union, List, Optional, Tuple
import anndata as ad
import numpy as np
import torch
from torch import cuda
import torch.nn as nn
from torch.nn import functional as F
from src.dataclasses import BatchData
from src.data import CrossSpeciesInferenceDataset
from src.utils import negative_binomial_loss
from src.modules import Encoder, Decoder, ParametricClusterer
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
        homology_dropout_rate: float = 0.2,
        base_learning_rate: float = 1e-3,
        base_batch_size: int = 256,
        batch_size: int = 256,
        min_learning_rate: float = 1e-5,
        warmup_data: float = 0.1,
        init_beta: float = 1e-3,
        final_beta: float = 0.1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        
        # Loss weights
        direct_recon_weight: float = 1.0,
        cross_species_recon_weight: float = 1.0,

        # Parametric Clusterer
        n_clusters: int = 100,
        initial_sigma: float = 1.0,
        cluster_warmup_epochs: int = 3,
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
        
        # Loss weights
        self.direct_recon_weight = direct_recon_weight
        self.cross_species_recon_weight = cross_species_recon_weight

        # Model parameters
        self.species_vocab_sizes = species_vocab_sizes
        self.n_species = len(species_vocab_sizes)
        self.n_latent = n_latent
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.homology_dropout_rate = homology_dropout_rate
        self.n_clusters = n_clusters
        self.initial_sigma = initial_sigma
        self.cluster_warmup_epochs = cluster_warmup_epochs

        # Initialize model components
        self._init_model_components()
        
        self.homology_available = homology_edges is not None
        if self.homology_available:
            self._init_homology_parameters(homology_edges, homology_scores)


    def _init_homology_parameters(self, homology_edges, homology_scores):        
        # Register edges as buffers
        for src_id, species_edges in homology_edges.items():
            for dst_id, edges in species_edges.items():
                self.register_buffer(
                    f'edges_{src_id}_{dst_id}',
                    edges if isinstance(edges, torch.Tensor) else torch.tensor(edges)
                )
        
        # Initialize learnable homology scores for each species pair
        for src_id, species_edges in homology_edges.items():
            for dst_id, edges in species_edges.items():
                if dst_id <= src_id:
                    continue

                init_scores = (
                    torch.ones(len(edges), dtype=torch.float32) if homology_scores is None
                    else homology_scores[src_id][dst_id] / homology_scores[src_id][dst_id].max()
                )
                # Register as parameter instead of buffer for learned homology
                self.register_parameter(
                    f'scores_{src_id}_{dst_id}',
                    nn.Parameter(init_scores)
                )
         

    def _init_model_components(self):
        self.mu_layer = nn.Linear(self.hidden_dims[-1], self.n_latent)
        self.logvar_layer = nn.Linear(self.hidden_dims[-1], self.n_latent)

        self.encoders = nn.ModuleDict({
            str(species_id): Encoder(
                n_genes=vocab_size,
                hidden_dims=self.hidden_dims,
                mu_layer=self.mu_layer,
                logvar_layer=self.logvar_layer,                
                dropout_rate=self.dropout_rate,
                n_latent=self.n_latent,
            )
            for species_id, vocab_size in self.species_vocab_sizes.items()
        })
    
        self.clusterers = nn.ModuleDict({
            str(species_id): ParametricClusterer(
                n_clusters=self.n_clusters,
                latent_dim=self.n_latent,
                initial_sigma=self.initial_sigma,
            )
            for species_id in self.species_vocab_sizes.keys()
        })     

        self.decoders = nn.ModuleDict({
            str(species_id): Decoder(
                n_genes=vocab_size,
                n_latent=self.n_latent,
                hidden_dims=self.hidden_dims[::-1],  # Reverse hidden dims for decoder
                num_clusters=self.n_clusters,
                dropout_rate=self.dropout_rate,
            )
            for species_id, vocab_size in self.species_vocab_sizes.items()
        })    

   

    def training_step(self, batch: BatchData, batch_idx: int):     
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"].detach(), sync_dist=True)
        self.log("train_direct_recon", loss_dict["direct_recon"].detach(), sync_dist=True)
        self.log("train_cross_species_recon", loss_dict["cross_species_recon"].detach(), sync_dist=True)
        self.log("train_direct_cluster_recon", loss_dict["direct_cluster_recon"].detach(), sync_dist=True)
        self.log("train_cross_species_cluster_recon", loss_dict["cross_species_cluster_recon"].detach(), sync_dist=True)
        self.log("train_kl", loss_dict["kl"].detach(), sync_dist=True)
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
            "val_direct_cluster_recon": loss_dict["direct_cluster_recon"].detach(),
            "val_cross_species_cluster_recon": loss_dict["cross_species_cluster_recon"].detach(),
            "val_kl": loss_dict["kl"].detach(),
        })
        
        return loss_dict["loss"]
    
    def on_train_epoch_start(self):
        """Reset correlation tracking at the start of each epoch."""        
        self.prev_gpu_memory = 0
        if cuda.is_available():
            cuda.empty_cache()
    

    def _compute_direct_reconstruction_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData,
        cluster_mode: bool = False,
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

            prefix = 'cluster_' if cluster_mode else ''

            pred_counts = torch.exp(reconstructions[target_species_id][f'{prefix}mean']) - 1
            pred_norm = pred_counts / pred_counts.sum(dim=1, keepdim=True) * 10_000            
            
            count_loss_nb += negative_binomial_loss(
                pred=pred_norm.clamp(min=1e-6),
                target=target_norm,
                theta=reconstructions[target_species_id][f'{prefix}theta'],
            )
            counter += 1
            
        return self.direct_recon_weight * count_loss_nb / counter
    
    def _compute_cross_species_reconstruction_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData,
        cluster_mode: bool = False,
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
                
                prefix = 'cluster_' if cluster_mode else ''
                pred_counts = torch.exp(recon[f'{prefix}mean']) - 1
                pred_norm = pred_counts / pred_counts.sum(dim=1, keepdim=True) * 10_000            
                
                count_loss_nb += negative_binomial_loss(
                    pred=pred_norm.clamp(min=1e-6),
                    target=target_norm,
                    theta=recon[f'{prefix}theta'],
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
            for _, encoder_output in encoder_outputs.items():
                
                mu = encoder_output['mu']
                logvar = encoder_output['logvar']
                
                # Add numerical stability clamps
                logvar = torch.clamp(logvar, min=-10, max=2)  # Prevent extreme values
                mu = torch.clamp(mu, min=-5, max=5)  # Prevent extreme values
            
                kl += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                counter += 1
                
        return kl / counter
    



    def compute_loss(self, batch: BatchData, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        direct_recon_loss = self._compute_direct_reconstruction_loss(outputs, batch, cluster_mode=False)

        cross_species_recon_loss = self._compute_cross_species_reconstruction_loss(outputs, batch, cluster_mode=False)

        direct_cluster_recon_loss = self._compute_direct_reconstruction_loss(outputs, batch, cluster_mode=True)
        cross_species_cluster_recon_loss = self._compute_cross_species_reconstruction_loss(outputs, batch, cluster_mode=True)

        kl_loss = self._compute_kl_loss(outputs)
        beta = self.get_current_beta()
        total_loss = direct_recon_loss + cross_species_recon_loss + direct_cluster_recon_loss + cross_species_cluster_recon_loss + beta * kl_loss
        
        return {
            "loss": total_loss,
            "direct_recon": direct_recon_loss,
            "cross_species_recon": cross_species_recon_loss,
            "direct_cluster_recon": direct_cluster_recon_loss,
            "cross_species_cluster_recon": cross_species_cluster_recon_loss,
            "kl": kl_loss,
        }
    
    # @torch.no_grad()
    def _transform_expression(
        self,
        batch: BatchData,
        src_species: int,
        dst_species: int,
    ) -> torch.Tensor:
        # Create dense transformation matrix
        edges = getattr(self, f'edges_{src_species}_{dst_species}')
        src, dst = sorted([src_species, dst_species])
        scores = getattr(self, f'scores_{src}_{dst}')
        
        # Apply dropout during training
        if self.training:
            scores = F.dropout(scores, p=self.homology_dropout_rate, training=True)
        
        n_src_genes = self.species_vocab_sizes[src_species]
        n_dst_genes = self.species_vocab_sizes[dst_species]
        transform_matrix = torch.zeros(
            n_dst_genes,
            n_src_genes,
            device=batch.data[src_species].device
        )
        transform_matrix[edges[:, 1], edges[:, 0]] = scores
        # Normalize transform matrix so src edges sum to 1.0 per dst gene
        transform_matrix = transform_matrix / (transform_matrix.sum(dim=1, keepdim=True).clamp(min=1))

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
            reconstructions = {}
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
            

            for target_species_id in self.species_vocab_sizes.keys():
                target_encoded = encoder_outputs[target_species_id]    

                memberships = self.clusterers[str(target_species_id)](target_encoded['z'])
                reconstructions[target_species_id] = source_decoder(target_encoded['z'], memberships)
            
            results[source_species_id] = {
                'encoder_outputs': encoder_outputs,
                'reconstructions': reconstructions,
            }
    

        return results

    @torch.no_grad()
    def get_latent_embeddings(
        self,
        species_data: Dict[str, Union[str, List[str], ad.AnnData, List[ad.AnnData]]],
        batch_size: int = 512,
        device: Optional[torch.device] = "cuda",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get latent embeddings with optional species-based correction.
        
        Args:
            species_data: Dictionary mapping species names to AnnData objects
            return_species: Whether to return species labels
            batch_size: Batch size for processing
            device: Device to use for computation
        """
        if device is None:
            device = next(self.parameters()).device
        elif device is not None:
            self.to(device)
            
        dataset = CrossSpeciesInferenceDataset(
            species_data=species_data,
            batch_size=batch_size,
        )
        
        self.eval()

        all_latents = []
        all_latents_z = []
        all_species = []
        
        # Collect latents and calculate species means
        for batch in dataset:
            batch = BatchData(
                data={k: v.to(device) for k,v in batch.data.items()},
            )
            for species_id in batch.data:
                encoded = self.encoders[str(species_id)](batch.data[species_id])
                latents = encoded['mu']
                latents_z = encoded['z']
                all_latents.append(latents.cpu())
                all_latents_z.append(latents_z.cpu())
                
                species_idx = torch.full((latents.shape[0],), species_id, dtype=torch.long)
                all_species.append(species_idx)
             

        # Concatenate all latents
        latents = torch.cat(all_latents, dim=0)
        latents_z = torch.cat(all_latents_z, dim=0)
        species = torch.cat(all_species, dim=0)
        
        memberships = {}
        for species_id in self.species_vocab_sizes.keys():
            mu_membership = self.clusterers[str(species_id)](latents.to(device))
            memberships[species_id] = mu_membership
        
        return latents, species, memberships
        
