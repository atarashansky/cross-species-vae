from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneImportanceModule(nn.Module):
    def __init__(
        self, 
        n_genes: int, 
        n_species: int,
        n_hidden: int = 128, 
        dropout: float = 0.1,
        deeply_inject_species: bool = False,        
    ):
        super().__init__()
        
        self.global_weights = nn.Parameter(torch.ones(n_genes))
        
        self.n_species = n_species if deeply_inject_species else 0
        
        self.deeply_inject_species = deeply_inject_species
        # Separate normalization for gene expressions
        self.gene_norm = nn.LayerNorm(n_genes)
        
        self.dropout = nn.Dropout(dropout)
        self.context_net_in = nn.Sequential(
            # First process concatenated input
            nn.Linear(n_genes + self.n_species, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            self.dropout
        )
        self.context_net_out = nn.Sequential(
            nn.Linear(n_hidden + self.n_species, n_genes),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, species_id: torch.Tensor | int | None = None) -> torch.Tensor:
        # Normalize only the gene expressions

        
        if species_id is not None and isinstance(species_id, int):
            species_id = torch.full((x.shape[0],), species_id, dtype=torch.long, device=x.device)
        
        species_one_hot = None
        if self.deeply_inject_species and species_id is not None:
            species_one_hot = F.one_hot(species_id, self.n_species)
        
        global_weights = torch.sigmoid(self.global_weights)

        x_normed = self.gene_norm(x)
        
        # Concatenate normalized gene expressions with categorical inputs
        context_input = x_normed
        if species_one_hot is not None:
            context_input = torch.cat([context_input, species_one_hot], dim=1)
        
        context_input = self.context_net_in(context_input)

        if species_one_hot is not None:
            context_input = torch.cat([context_input, species_one_hot], dim=1)
        
        # Apply importance weighting
        context_weights = self.context_net_out(context_input)
        importance = global_weights * context_weights
        
        return x * importance


class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_species: int,
        mu_layer: nn.Linear,
        logvar_layer: nn.Linear,
        hidden_dims: list,
        n_context_hidden: int = 128,
        dropout_rate: float = 0.1,
        deeply_inject_species: bool = False,
    ):
        super().__init__()
        
        self.n_species = n_species if deeply_inject_species else 0
        self.deeply_inject_species = deeply_inject_species
        self.gene_importance = GeneImportanceModule(
            n_genes, 
            self.n_species, 
            n_hidden = n_context_hidden,
            deeply_inject_species = deeply_inject_species,             
        )        
        
        # Get input dimension for encoder layers
        self.encoder = self._make_encoder_layers(n_genes, hidden_dims, dropout_rate)
        self.mu = mu_layer
        self.logvar = logvar_layer

    def _make_encoder_layers(self, input_dim: int, hidden_dims: list, dropout_rate: float) -> nn.ModuleList:
        layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            # Each layer block now takes species information into account
            layer_block = nn.ModuleDict({
                'linear': nn.Linear(dims[i] + self.n_species, dims[i + 1]),
                'norm': nn.LayerNorm(dims[i + 1]),
                'relu': nn.ReLU(),
                'dropout': nn.Dropout(dropout_rate)
            })
            layers.append(layer_block)
        return layers

    def preprocess(self, x: torch.Tensor, species_id: torch.Tensor | int | None = None) -> torch.Tensor:
        """Apply gene importance weighting"""
        return x + self.gene_importance(x, species_id)
    
    def embed(self, x: torch.Tensor, species_id: torch.Tensor | int | None = None) -> torch.Tensor:
        """Transform input to embedding space"""
        return self.preprocess(x, species_id)
    
    def encode(self, embedded: torch.Tensor, species_id: torch.Tensor | int | None = None) -> Dict[str, torch.Tensor]:
        """Process embedded input through encoder layers"""
        # Create one-hot encoding for species
        if isinstance(species_id, int) and species_id is not None:
            species_id = torch.full((embedded.shape[0],), species_id, dtype=torch.long, device=embedded.device)
        
        species_one_hot = None
        if self.deeply_inject_species and species_id is not None:
            species_one_hot = F.one_hot(
                species_id, 
                self.n_species
            )
        
        # Process through encoder layers
        h = embedded
        for layer_block in self.encoder:
            # Concatenate species information at each layer
            if species_one_hot is not None:
                layer_input = torch.cat([h, species_one_hot], dim=1)
            else:
                layer_input = h

            h = layer_block['linear'](layer_input)
            h = layer_block['norm'](h)
            h = layer_block['relu'](h)
            h = layer_block['dropout'](h)
        
        # Final concatenation for mu and logvar
        if species_one_hot is not None:
            final_h = torch.cat([h, species_one_hot], dim=1)
        else:
            final_h = h
        
        mu = torch.clamp(self.mu(final_h), min=-20, max=20)
        logvar = torch.clamp(self.logvar(final_h), min=-20, max=20)
        z = self.reparameterize(mu, logvar)
        return {'mu': mu, 'logvar': logvar, 'z': z}
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution"""
        logvar = torch.clamp(logvar, min=-20, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std if self.training else mu
    
    def forward(self, x: torch.Tensor, species_id: torch.Tensor | int) -> Dict[str, torch.Tensor]:
        embedded = self.embed(x, species_id)
        return self.encode(embedded, species_id)

class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_latent: int,
        hidden_dims: list,
        deeply_inject_species: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.n_species = n_species if deeply_inject_species else 0
        self.deeply_inject_species = deeply_inject_species

        # Build decoder network
        def build_decoder_layers(dims: list) -> nn.ModuleList:
            layers = nn.ModuleList()
            
            for i in range(len(dims) - 1):
                is_last_layer = i == len(dims) - 2
                
                layer_block = nn.ModuleDict({
                    'linear': nn.Linear(dims[i] + self.n_species, dims[i + 1]),
                    'norm': nn.LayerNorm(dims[i + 1]) if not is_last_layer else nn.Identity(),
                    'activation': nn.ReLU() if not is_last_layer else nn.Softplus(),
                    'dropout': nn.Dropout(dropout_rate) if not is_last_layer else nn.Identity()
                })
                layers.append(layer_block)
            
            return layers
        
        dims = [n_latent] + hidden_dims + [n_genes]
        
        self.decoder_net = build_decoder_layers(dims)
        self.theta_net = build_decoder_layers(dims)
    
    def _process_through_layers(self, z: torch.Tensor, layers: nn.ModuleList, species_id: torch.Tensor | int) -> torch.Tensor:
        # Create one-hot encoding for species
        if isinstance(species_id, int) and species_id is not None:
            species_id = torch.full((z.shape[0],), species_id, dtype=torch.long, device=z.device)
        
        species_one_hot = None
        if self.deeply_inject_species and species_id is not None:
            species_one_hot = F.one_hot(
                species_id, 
                self.n_species
            )
        
        h = z
        for layer_block in layers:
            # Concatenate species information at each layer
            if species_one_hot is not None:
                layer_input = torch.cat([h, species_one_hot], dim=1)
            else:
                layer_input = h

            h = layer_block['linear'](layer_input)
            h = layer_block['norm'](h)
            h = layer_block['activation'](h)
            h = layer_block['dropout'](h)
        
        return h
    
    def forward(self, z: torch.Tensor, species_id: torch.Tensor | int | None = None) -> Dict[str, torch.Tensor]:
        return {
            'mean': self._process_through_layers(z, self.decoder_net, species_id),
            'theta': torch.exp(self._process_through_layers(z, self.theta_net, species_id))
        }
    