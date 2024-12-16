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

class FrozenEmbedding(nn.Module):
    def __init__(self, embedding_weights: torch.Tensor):
        super().__init__()

        with torch.no_grad():
            self.linear = nn.Linear(embedding_weights.shape[0], embedding_weights.shape[1], bias=False)
            self.linear.weight.copy_(embedding_weights.T)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,   
        hidden_dims: list,
        mu_layer: nn.Linear | None = None,
        logvar_layer: nn.Linear | None = None,        
        n_context_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_latent: int = 128,
    ):
        super().__init__()
        
        self.gene_importance = GeneImportanceModule(n_genes, n_context_hidden)        
        
        # Get input dimension for encoder layers
        self.encoder = self._make_encoder_layers(n_genes, hidden_dims, dropout_rate)
        if mu_layer is not None:
            self.mu = mu_layer
        else:
            self.mu = nn.Linear(hidden_dims[-1], n_latent)

        if logvar_layer is not None:
            self.logvar = logvar_layer
        else:
            self.logvar = nn.Linear(hidden_dims[-1], n_latent)


    @staticmethod
    def _make_encoder_layers(input_dim: int, hidden_dims: list, dropout_rate: float) -> nn.Sequential:
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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gene importance weighting"""            
        return x + self.gene_importance(x)
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input to embedding space"""
        return self.preprocess(x)
    
    def encode(self, embedded: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process embedded input through encoder layers"""
        h = self.encoder(embedded)
        mu = torch.clamp(self.mu(h), min=-20, max=20)
        logvar = torch.clamp(self.logvar(h), min=-20, max=20)
        z = self.reparameterize(mu, logvar)
        return {'mu': mu, 'logvar': logvar, 'z': z}
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution"""
        logvar = torch.clamp(logvar, min=-20, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std if self.training else mu
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedded = self.embed(x)
        return self.encode(embedded)

class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        hidden_dims: list,
        dropout_rate: float = 0.1,
    ):
        super().__init__()


        # Build decoder network
        def build_decoder_layers(dims: list) -> nn.Sequential:
            layers = []
            
            for i in range(len(dims) - 1):
                is_last_layer = i == len(dims) - 2
            
                layers.extend([
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]) if not is_last_layer else nn.Identity(),
                    nn.ReLU() if not is_last_layer else nn.Softplus(),
                    nn.Dropout(dropout_rate) if not is_last_layer else nn.Identity()
                ])
            
            return nn.Sequential(*layers)
        
        dims = [n_latent] + hidden_dims + [n_genes]
        
        self.decoder_net = build_decoder_layers(dims)
        self.theta_net = build_decoder_layers(dims)


    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'mean': self.decoder_net(z),
            'theta': torch.exp(self.theta_net(z))
        }
    