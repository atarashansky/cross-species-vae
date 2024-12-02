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
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x + self.gene_importance(x))
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
        hidden_dims: list,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.log_theta = nn.Parameter(torch.ones(n_genes) * 2.3)

        # Technical scaling network
        self.scaling_net = nn.Sequential(
            nn.Linear(n_latent, n_genes),
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
            
        self.decoder_net = nn.Sequential(*layers)     
    
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get biological factors
        bio_factors = self.decoder_net(z)
        scaling_factors = self.scaling_net(z)
        mean = bio_factors * scaling_factors
        theta = torch.exp(self.log_theta)  
        
        return {
            'mean': mean,
            'theta': theta
        }
    

class CrossSpeciesVAE(pl.LightningModule):
    """Cross-species VAE with multi-scale encoding and species-specific components."""

    STAGE_MAPPING = {
        0: "transform_recon",
        1: "homology_loss",
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
        warmup_epochs: float = 0.1,
        init_beta: float = 1e-3,
        final_beta: float = 0.1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        recon_weight: float = 1.0,
        homology_weight: float = 1.0,
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

        self.last_homology_loss = 0.0
        self.last_recon_loss = 0.0
        self.last_kl_loss = 0.0
        


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
        for key, value in loss_dict.items():
            log_key = f"train_{key}"
            if value == 0.0:
                value = getattr(self, f"last_{key}_loss")

            value_to_log = value.detach() if torch.is_tensor(value) else value
            self.log(log_key, value_to_log, sync_dist=True)
        
        return loss_dict["loss"]
        
    def validation_step(self, batch: BatchData, batch_idx: int):
        """Validation step."""
        # Forward pass
        outputs = self(batch)
        
        # Compute losses
        loss_dict = self.compute_loss(batch, outputs)
        
        # Store only the scalar values, detached from computation graph
        log_loss_dict = {}
        for key, value in loss_dict.items():
            log_key = f"val_{key}"
            if value == 0.0:
                value = getattr(self, f"last_{key}_loss")
            # Convert all values to tensors
            value_to_log = torch.tensor(value, device=self.device) if isinstance(value, float) else value.detach()
            log_loss_dict[log_key] = value_to_log
            
        self.validation_step_outputs.append(log_loss_dict)
        
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

        print(f"Current epoch: {self.trainer.current_epoch}", "Current stage: ", self.get_stage())
        
        
   
    def _compute_homology_loss(self, outputs: Dict[int, Any], batch: BatchData) -> torch.Tensor:   
        if self.get_stage() != "homology_loss":
            return torch.tensor(0.0, device=self.device)
        
        assembled_data = {}
        for target_species_id in outputs:
            # For each target species, collect real and transformed data
            data_to_concat = []
            
            # Add data from each source species (including self)
            for source_species_id in self.species_vocab_sizes.keys():
                if source_species_id == target_species_id:
                    # Use real data for matching species
                    data_to_concat.append(batch.data[source_species_id])
                else:
                    # Use reconstructed data for other species
                    recon = outputs[source_species_id]['reconstructions'][target_species_id]
                    data_to_concat.append(recon['mean'])
            
            # Concatenate all data for this target species space
            assembled_data[target_species_id] = torch.cat(data_to_concat, dim=0)
        
        homology_loss = torch.tensor(0.0, device=self.device)
        
        # Compute homology loss across species pairs
        for src_species_id in assembled_data:
            for dst_species_id in assembled_data:
                if src_species_id == dst_species_id:
                    continue
                
                # Get edges and scores for this species pair
                edges = self.homology_edges[src_species_id][dst_species_id]
                scores = torch.sigmoid(self.homology_scores[str(src_species_id)][str(dst_species_id)])

                src, dst = edges.t()
                src_pred = assembled_data[src_species_id][:, src]
                dst_pred = assembled_data[dst_species_id][:, dst]

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

        loss = self.homology_weight * homology_loss / (len(self.species_vocab_sizes) - 1) / len(self.species_vocab_sizes)
        self.last_homology_loss = loss
        return loss


    def _compute_reconstruction_loss(
        self,
        outputs: Dict[str, Any],
        batch: BatchData,
    ) -> torch.Tensor:
        if self.get_stage() not in ["direct_recon", "transform_recon"]:
            return torch.tensor(0.0, device=self.device)
        
        count_loss_nb = torch.tensor(0.0, device=self.device)
        for target_species_id in outputs:
            reconstructions = outputs[target_species_id]['reconstructions']
            
            target_counts = batch.data[target_species_id]
            target_counts = torch.exp(target_counts) - 1
            nonzero_fraction = (target_counts > 0).float().mean()        
            target_norm = target_counts / target_counts.sum(dim=1, keepdim=True) * 10_000                  

            for _, recon in reconstructions.items():
                pred_counts = torch.exp(recon['mean']) - 1
                pred_norm = pred_counts / pred_counts.sum(dim=1, keepdim=True) * 10_000            
                
                count_loss_nb += negative_binomial_loss(
                    pred=pred_norm.clamp(min=1e-6),
                    target=target_norm,
                    theta=recon['theta'],
                )
    
        loss = self.recon_weight * count_loss_nb * (1 - nonzero_fraction) / self.n_species
        self.last_recon_loss = loss
        return loss
    
    def _compute_kl_loss(
        self,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        if self.get_stage() not in ["direct_recon", "transform_recon"]:
            return torch.tensor(0.0, device=self.device)
        
        kl = torch.tensor(0.0, device=self.device)
        for target_species_id in outputs:
            # Get encoder outputs for all species
            encoder_outputs = outputs[target_species_id]['encoder_outputs']
            
            # Compute KL for each encoding
            for _, enc_output in encoder_outputs.items():
                mu = enc_output['mu']
                logvar = enc_output['logvar']
                kl += -0.5 * torch.mean(1 + logvar - mu.pow(2).clamp(max=100) - logvar.exp().clamp(max=100))
                
        loss = kl / self.n_species
        self.last_kl_loss = loss
        return loss
    


    def compute_loss(self, batch: BatchData, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        recon_loss = self._compute_reconstruction_loss(outputs, batch)
        
        kl = self._compute_kl_loss(outputs)

        homology_loss = self._compute_homology_loss(outputs, batch)
        

        beta = self.get_current_beta()
        total_loss = recon_loss + kl * beta + homology_loss
        
        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl,
            "homology": homology_loss,
        }
    
    
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
            device=batch.data[src_species].device
        )
        transform_matrix[edges[:, 1], edges[:, 0]] = scores
        # Normalize transform matrix so src edges sum to 1.0 per dst gene
        transform_matrix = transform_matrix / (transform_matrix.sum(dim=1, keepdim=True) + 1e-8)

        x = batch.data[src_species]

        # Transform expression data
        transformed = x @ transform_matrix.t()
        return transformed

    """
    What worked (refer to main branch for a working implementation!!):
    - Transform reconstruction (A vs A, A-->B vs B, A-->C vs C)
    The decoder had two networks: multi-species and single-species.
     - The multi-species network was used for the transform reconstruction. Multi-species network concatenated all latents (A, A-->B, A-->C) and decoded with the target decoders.
     - The single-species network was used for the homology loss.
    """
    # TODO: This could be another stage but idk if it matters.
    # def _transform_recon_forward_old(self, batch: BatchData) -> Dict[str, Any]:
    #     """Handle transform reconstruction stage.
        
    #     For each source species:
    #         1. Transform source data to all target spaces (A->B, A->C)
    #         2. Encode all data (original and transformed) with respective encoders
    #         3. Concatenate all latents and decode with each target decoder
    #         4. Compare against original/transformed data
    #     """
    #     results = {}
        
    #     for source_species_id, source_data in batch.data.items():
    #         # For each source species (A, B, C)            
    #         reconstructions = {}
    #         encoder_outputs = {}
                        
    #         source_encoder = self.encoders[str(source_species_id)]
    #         source_encoded = source_encoder(source_data)
    #         encoder_outputs[source_species_id] = source_encoded

    #         inputs = {source_species_id: source_data}
            
    #         for target_species_id in self.species_vocab_sizes.keys():
    #             if target_species_id == source_species_id:
    #                 continue
                    
    #             # Transform source data to target space
    #             transformed = self._transform_expression(batch, source_species_id, target_species_id)
    #             inputs[target_species_id] = transformed
                
    #             # Encode transformed data with target encoder
    #             target_encoder = self.encoders[str(target_species_id)]
    #             transformed_encoded = target_encoder(transformed)
    #             encoder_outputs[target_species_id] = transformed_encoded
            
    #         # Now decode using concatenated latents for each target space
    #         for target_species_id in self.species_vocab_sizes.keys():
    #             target_decoder = self.decoders[str(target_species_id)]

    #             # Decode with target decoder using all latents
    #             reconstructions[target_species_id] = target_decoder(encoder_outputs[target_species_id]['z'])
            
    #         results[source_species_id] = {
    #             'encoder_outputs': encoder_outputs,
    #             'reconstructions': reconstructions,
    #             'inputs': inputs,
    #         }
        
    #     return results
    
    def _transform_recon_forward(self, batch: BatchData) -> Dict[str, Any]:
        """Handle transform reconstruction stage.
        
        For each source species:
            1. Transform source data to all target spaces (A->B, A->C)
            2. Encode all data (original and transformed) with respective encoders
            3. Concatenate all latents and decode with each target decoder
            4. Compare against original/transformed data
        """
        results = {}
        
        for source_species_id, source_data in batch.data.items():
            # For each source species (A, B, C)            
            reconstructions = {}
            encoder_outputs = {}
                        
            source_encoder = self.encoders[str(source_species_id)]
            source_encoded = source_encoder(source_data)
            encoder_outputs[source_species_id] = source_encoded
            
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
            
            results[source_species_id] = {
                'encoder_outputs': encoder_outputs,
                'reconstructions': reconstructions,
            }
        
        return results
        
    def _homology_loss_forward(self, batch: BatchData) -> Dict[str, Any]:
        """Handle homology loss stage."""
        results = {}
        
        # First encode all species data
        encoded_species = {}
        for species_id, data in batch.data.items():
            encoded_species[species_id] = self.encoders[str(species_id)](data)
        
        # For each source species, decode into all target spaces
        for src_species_id in batch.data:
            
            reconstructions = {}
            for dst_species_id in self.species_vocab_sizes.keys():
                reconstructions[dst_species_id] = self.decoders[str(dst_species_id)](encoded_species[src_species_id]['z'])
            
            results[src_species_id] = {
                'encoder_outputs': {src_species_id: encoded_species[src_species_id]},
                'reconstructions': reconstructions,
            }
        
        return results

    def forward(self, batch: BatchData) -> Dict[str, Any]:
        """Main forward pass handling different stages."""
        if self.get_stage() == "transform_recon":
            return self._transform_recon_forward(batch)
        elif self.get_stage() == "homology_loss":
            return self._homology_loss_forward(batch)
        else:
            raise ValueError(f"Unknown stage: {self.get_stage()}")

    def get_stage(self):
        return self.STAGE_MAPPING[self.trainer.current_epoch % 2]
    
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
