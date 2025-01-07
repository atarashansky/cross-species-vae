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
from src.modules import Encoder, Decoder, ParametricClusterer, ESMGeneAggregator
from src.base_model import BaseModel


class CrossSpeciesVAE(BaseModel):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    def __init__(
        self,
        species_vocab_sizes: Dict[int, int],
        gene_embeddings: Dict[int, torch.Tensor],
        n_latent: int = 256,
        hidden_dims: list = [256],
        dropout_rate: float = 0.2,
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
        recon_weight: float = 1.0,

        # Parametric Clusterer
        n_clusters: int = 100,
        cluster_warmup_epochs: int = 3,
        initial_alpha: float = 0.25,

        aggregator_dim: int = 512,
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

        self.save_hyperparameters(ignore=['gene_embeddings'])
        
        # Loss weights
        self.recon_weight = recon_weight

        # Model parameters
        self.species_vocab_sizes = species_vocab_sizes
        self.n_species = len(species_vocab_sizes)
        self.n_latent = n_latent
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.n_clusters = n_clusters
        self.cluster_warmup_epochs = cluster_warmup_epochs
        self.initial_alpha = initial_alpha
        self.aggregator_dim = aggregator_dim

        self.gene_embeddings = gene_embeddings

        # Initialize model components
        self._init_model_components()
        

    def _init_model_components(self):
        self.esm_aggregators = nn.ModuleDict({
            str(species_id): ESMGeneAggregator(
                n_genes=vocab_size, 
                gene_embeddings=self.gene_embeddings[species_id],
                dropout=self.dropout_rate,
                aggregator_hidden_dim=self.aggregator_dim,
                aggregator_output_dim=self.aggregator_dim,
            )
            for species_id, vocab_size in self.species_vocab_sizes.items()
        })

        self.encoder = Encoder(
            input_dim=self.aggregator_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            n_latent=self.n_latent,
        )
    
        self.clusterer = ParametricClusterer(
            n_clusters=self.n_clusters,
            latent_dim=self.n_latent,
        )

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

        self.log_alpha = nn.Parameter(torch.log(torch.tensor(self.initial_alpha)))

   

    def training_step(self, batch: BatchData, batch_idx: int):     
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        
        # Log metrics
        self.log("train_loss", loss_dict["loss"].detach(), sync_dist=True)
        self.log("train_recon", loss_dict["recon"].detach(), sync_dist=True)
        self.log("train_kl", loss_dict["kl"].detach(), sync_dist=True)
        self.log("train_alpha", self.get_current_alpha().detach(), sync_dist=True)
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
            "val_alpha": self.get_current_alpha().detach(),
        })
        
        return loss_dict["loss"]
    
    def on_train_epoch_start(self):
        """Reset correlation tracking at the start of each epoch."""        
        self.prev_gpu_memory = 0
        if cuda.is_available():
            cuda.empty_cache()
   
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        
        if self.current_epoch == self.cluster_warmup_epochs - 1:
            z = []
            train_dataloader = self.trainer.train_dataloader
            
            for batch in train_dataloader: 
                for species_id, data in batch.data.items():
                    with torch.no_grad():
                        data = data.to(self.device)
                        latents_mu = self.encode(data, species_id)['mu']
                        z.append(latents_mu)

            z = torch.cat(z, dim=0)
            self.clusterer.initialize_with_kmeans(z)


    def _compute_reconstruction_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData,
    ) -> torch.Tensor:
    
        count_loss_nb = torch.tensor(0.0, device=self.device)

        if self.recon_weight == 0.0:
            return count_loss_nb
        
        counter = 0
        
        for target_species_id in outputs:
            reconstructions = outputs[target_species_id]['reconstruction']
            
            target_counts = batch.data[target_species_id]
            target_counts = torch.exp(target_counts) - 1
            target_norm = target_counts / target_counts.sum(dim=1, keepdim=True) * 10_000                  


            pred_counts = torch.exp(reconstructions['mean']) - 1
            pred_norm = pred_counts / pred_counts.sum(dim=1, keepdim=True) * 10_000            
            
            count_loss_nb += negative_binomial_loss(
                pred=pred_norm.clamp(min=1e-6),
                target=target_norm,
                theta=reconstructions['theta'],
            )
            counter += 1
            
        return self.recon_weight * count_loss_nb / counter
    
    def _compute_kl_loss(
        self,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        kl = torch.tensor(0.0, device=self.device)
        counter = 0
        for target_species_id in outputs:
            # Get encoder outputs for all species
            encoder_output = outputs[target_species_id]['encoder_output']
                
            mu = encoder_output['mu']
            logvar = encoder_output['logvar']
            
            # Add numerical stability clamps
            logvar = torch.clamp(logvar, min=-10, max=2)  # Prevent extreme values
            mu = torch.clamp(mu, min=-5, max=5)  # Prevent extreme values
        
            kl += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            counter += 1
                
        return kl / counter
    

    def compute_loss(self, batch: BatchData, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        recon_loss = self._compute_reconstruction_loss(outputs, batch)
        kl_loss = self._compute_kl_loss(outputs)
        beta = self.get_current_beta()
        total_loss = recon_loss + beta * kl_loss
        
        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
        }
    
    def encode(self, data: torch.Tensor, species_id: int) -> Dict[str, Any]:
        aggregated = self.esm_aggregators[str(species_id)](data)
        encoded = self.encoder(aggregated)
        return encoded

    def forward(self, batch: BatchData) -> Dict[str, Any]:
        results = {}        
        for species_id, data in batch.data.items():
            encoded = self.encode(data, species_id)
                          
            decoder = self.decoders[str(species_id)]
            memberships = self.get_membership_scores(encoded)
            reconstruction = decoder(encoded['z'], memberships)

            results[species_id] = {
                'encoder_output': encoded,
                'reconstruction': reconstruction,
            }

        return results

    def get_current_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha)
    
    def get_membership_scores(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        alpha = self.get_current_alpha()
        return self.clusterer((1 - alpha) * encoded['z'] + alpha * encoded['mu'])        
    
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
                encoded = self.encode(batch.data[species_id], species_id)
                latents = encoded['mu']
                latents_z = encoded['z']
                all_latents.append(latents)
                all_latents_z.append(latents_z)
                
                species_idx = torch.full((latents.shape[0],), species_id, dtype=torch.long)
                all_species.append(species_idx)
             

        # Concatenate all latents
        latents = torch.cat(all_latents, dim=0)
        latents_z = torch.cat(all_latents_z, dim=0)
        species = torch.cat(all_species, dim=0)
        
        encoded = {
            'z': latents_z,
            'mu': latents,
        }
        memberships = self.get_membership_scores(encoded)
        
        return latents.cpu(), species.cpu(), memberships.cpu()
        
