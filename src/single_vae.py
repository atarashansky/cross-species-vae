from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

from src.utils import negative_binomial_loss
from src.modules import Encoder, Decoder
from src.multi_vae import CrossSpeciesVAE
from src.dataclasses import BatchData
from src.data import CrossSpeciesInferenceDataset
from src.base_model import BaseModel

class VAE(BaseModel):
    def __init__(
        self,
        cross_species_vae: CrossSpeciesVAE,
        n_latent: int = 256,
        hidden_dims: list = [256],
        dropout_rate: float = 0.2,
        base_learning_rate: float = 1e-3,
        base_batch_size: int = 256,
        batch_size: int = 256,
        min_learning_rate: float = 1e-5,
        init_beta: float = 1e-3,
        final_beta: float = 1.0,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        warmup_data: float = 0.1,
        deeply_inject_species: bool = False,
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

        self.save_hyperparameters(ignore=['cross_species_vae'])
        
        # Store and freeze the VAE
        self.cross_species_vae = cross_species_vae
        self.cross_species_vae.eval()
        for param in self.cross_species_vae.parameters():
            param.requires_grad = False
            
        # Calculate input dimension (sum of all species gene spaces)
        self.input_dim = sum(vocab_size for vocab_size in cross_species_vae.species_vocab_sizes.values())
        n_species = len(cross_species_vae.species_vocab_sizes)
        
        self.learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.base_batch_size = base_batch_size
        

        n_species_for_layer = n_species if deeply_inject_species else 0
        self.mu_layer = nn.Linear(hidden_dims[-1] + n_species_for_layer, n_latent)
        self.logvar_layer = nn.Linear(hidden_dims[-1] + n_species_for_layer, n_latent)
     
        self.encoder = Encoder(
            n_genes=self.input_dim,
            n_species=n_species,
            mu_layer=self.mu_layer,
            logvar_layer=self.logvar_layer,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            deeply_inject_species=deeply_inject_species,
        )   

        self.decoder = Decoder(
            n_genes=self.input_dim,
            n_species=n_species,
            n_latent=n_latent,
            hidden_dims=hidden_dims[::-1],  # Reverse hidden dims for decoder
            dropout_rate=dropout_rate,
            deeply_inject_species=deeply_inject_species,
        )
        
        self.validation_step_outputs = []
        
    def preprocess_batch(self, batch: BatchData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform batch through CrossSpeciesVAE and concatenate outputs."""
        with torch.no_grad():
            # For each source species
            all_reconstructions = []
            all_species_ids = []
            
            for source_species_id, source_data in batch.data.items():
                # Encode source data
                source_encoder = self.cross_species_vae.encoders[str(source_species_id)]
                source_encoded = source_encoder(source_data, source_species_id)
                                
                # Decode into all species spaces
                cell_reconstructions = []
                for target_species_id in sorted(self.cross_species_vae.species_vocab_sizes.keys()):
                    target_decoder = self.cross_species_vae.decoders[str(target_species_id)]
                    reconstruction = target_decoder(source_encoded['z'], target_species_id)
                    cell_reconstructions.append(reconstruction['mean'])
                
                # Concatenate all target species reconstructions
                concatenated = torch.cat(cell_reconstructions, dim=1)
                all_reconstructions.append(concatenated)
                all_species_ids.append(torch.full((concatenated.shape[0], ), source_species_id, dtype=torch.long, device=self.device))
                        
            return torch.cat(all_reconstructions, dim=0), torch.cat(all_species_ids, dim=0)
    
    def encode(self, x: torch.Tensor, species_id: torch.Tensor | int) -> Dict[str, torch.Tensor]:
        return self.encoder(x, species_id)
        
    def decode(self, z: torch.Tensor, species_id: torch.Tensor | int) -> Dict[str, torch.Tensor]:
        return self.decoder(z, species_id)
        
    def forward(self, x: torch.Tensor, species_id: torch.Tensor | int) -> Dict[str, torch.Tensor]:
        # Full forward pass
        encoded = self.encode(x, species_id)
        decoded = self.decode(encoded['z'], species_id)
        
        return {
            **encoded,
            **decoded
        }
    
    def _compute_reconstruction_loss(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        return negative_binomial_loss(
            pred=mean,
            target=x,
            theta=theta
        )
    
    def _compute_kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    
    def training_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        # Preprocess batch through frozen VAE
        x, species_ids = self.preprocess_batch(batch)
        
        # Forward pass through SCVI
        outputs = self(x, species_ids)
        
        # Compute losses
        recon_loss = self._compute_reconstruction_loss(
            x=x,
            mean=outputs['mean'],
            theta=outputs['theta']
        )
        
        kl_loss = self._compute_kl_loss(
            mu=outputs['mu'],
            logvar=outputs['logvar']
        )
        
        # Get current beta for KL weight
        beta = self.get_current_beta()
        
        # Compute losses with current beta
        total_loss = recon_loss + beta * kl_loss
        
        # Log metrics
        self.log('train_loss', total_loss)
        self.log('train_recon', recon_loss)
        self.log('train_kl', kl_loss)
        
        return total_loss
    
    def validation_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        # Preprocess batch through frozen VAE
        x, species_ids = self.preprocess_batch(batch)
        
        # Forward pass through SCVI
        outputs = self(x, species_ids)
        
        # Compute losses
        recon_loss = self._compute_reconstruction_loss(
            x=x,
            mean=outputs['mean'],
            theta=outputs['theta']
        )
        
        kl_loss = self._compute_kl_loss(
            mu=outputs['mu'],
            logvar=outputs['logvar']
        )
        
        # Total loss
        total_loss = recon_loss + self.get_current_beta() * kl_loss
        
        # Store outputs for epoch end averaging
        self.validation_step_outputs.append({
            "val_loss": total_loss.detach(),
            "val_recon": recon_loss.detach(),
            "val_kl": kl_loss.detach(),
        })
        
        return total_loss
    

    @torch.no_grad()
    def get_latent_embeddings(
        self,
        species_data,
        batch_size: int = 512,
        device = "cuda",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get latent embeddings for cells in provided AnnData objects.
        
        Args:
            species_data: Dictionary mapping species names to AnnData objects
            batch_size: Batch size for processing
            device: Device to use for computation. If None, uses model's device
            
        Returns:
            latents: Tensor of shape [n_cells, n_latent]
        """
        if device is None:
            device = next(self.parameters()).device
        elif device is not None:
            self.to(device)
            
        dataset = CrossSpeciesInferenceDataset(
            species_data=species_data,
            batch_size=batch_size,
        )
        
        # Set to evaluation mode
        self.eval()
        all_latents = []
        all_species_ids = []
        for batch in dataset:
            # Move batch to device and preprocess through VAE
            batch = BatchData(
                data={k: v.to(device) for k,v in batch.data.items()},
            )
            x, species_ids = self.preprocess_batch(batch)
            
            # Get latent representation from encoder
            encoded = self.encode(x, species_ids)
            all_latents.append(encoded['z'].cpu())
            all_species_ids.append(species_ids.cpu())        
        
        # Concatenate all batches
        latents = torch.cat(all_latents, dim=0)
        species_ids = torch.cat(all_species_ids, dim=0)
        return latents, species_ids