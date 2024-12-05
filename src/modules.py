from typing import Dict
import torch
import torch.nn as nn

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

        mu = torch.clamp(mu, min=-20, max=20)
        logvar = torch.clamp(logvar, min=-20, max=20)
        
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
        return {
            'mean': self.decoder_net(z),
            'theta': torch.exp(self.log_theta)
        }
    