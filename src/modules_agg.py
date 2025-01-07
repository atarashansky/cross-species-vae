from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


class ESMGeneAggregator(nn.Module):
    """
    A learnable aggregator that transforms a cell's gene-expression values x
    into an aggregated embedding v, using each gene's ESM embedding.
    
    Args:
        n_genes (int): Number of genes for the species.
        embed_dim (int): Dimensionality of each gene's ESM embedding.
        aggregator_hidden_dim (int): Hidden dimension for the aggregator MLP.
        aggregator_output_dim (int): Final dimension of the aggregated vector v.
        dropout (float): Dropout probability in the aggregator MLP.
        init_embeddings (torch.Tensor, optional): 
            Pre-computed ESM embeddings of shape (n_genes, embed_dim). 
            If None, will initialize randomly (not recommended for real usage).
    """
    def __init__(
        self,
        n_genes: int,
        gene_embeddings: torch.Tensor,
        aggregator_hidden_dim: int = 128,
        aggregator_output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # TODO: Consider freezing this
        self.register_buffer('gene_embeddings', gene_embeddings)
        embed_dim = gene_embeddings.shape[-1]

        self.gene_importance = GeneImportanceModule(n_genes)        
        
        # Simple MLP for per-gene transform
        self.aggregator = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, aggregator_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aggregator_hidden_dim, aggregator_output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): A batch of gene-expression counts or normalized values
                with shape (batch_size, n_genes).

        Returns:
            torch.Tensor: The aggregated embedding vector v for each cell,
                shape (batch_size, aggregator_output_dim).
        """
        # x shape: (B, n_genes)
        B = x.shape[0]
        n_genes, embed_dim = self.gene_embeddings.shape
        
        # Broadcast the gene embedding matrix so it matches batch size
        # shape: (B, n_genes, embed_dim)
        expanded_embeddings = self.gene_embeddings.unsqueeze(0).expand(B, -1, -1)
        
        x = self.gene_importance(x)

        # Reshape x to (B, n_genes, 1) so we can multiply each gene's expression by its embedding
        x_expanded = x.unsqueeze(-1)  # (B, n_genes, 1)
        
        # Weighted gene embeddings: (B, n_genes, embed_dim)
        weighted_embeddings = x_expanded * expanded_embeddings
        
        # Flatten so MLP processes each gene's embedding separately:
        # shape => (B * n_genes, embed_dim)
        flattened = weighted_embeddings.reshape(-1, embed_dim)
        
        # Pass each gene embedding through the aggregator MLP
        # shape => (B * n_genes, aggregator_output_dim)
        
        transformed = self.aggregator(flattened)
        
        # Reshape back to (B, n_genes, aggregator_output_dim)
        transformed = transformed.reshape(B, n_genes, -1)
        
        # Sum (or average) over the gene dimension => (B, aggregator_output_dim)
        v = transformed.mean(dim=1)
        # If you prefer to average, you can do:
        # v = transformed.mean(dim=1)
        
        return v


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,   
        hidden_dims: list,     
        dropout_rate: float = 0.1,
        n_latent: int = 128,
    ):
        super().__init__()
        
        self.encoder = self._make_encoder_layers(input_dim, hidden_dims, dropout_rate)
        self.mu = nn.Linear(hidden_dims[-1], n_latent)
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
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encode(x)
    
    
class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        hidden_dims: list,
        num_clusters: int,
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
        
        dims = [n_latent + num_clusters] + hidden_dims + [n_genes]
        
        self.decoder_net = build_decoder_layers(dims)
        self.theta_net = build_decoder_layers(dims)

    
    def forward(self, z: torch.Tensor, memberships: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = torch.cat([z, memberships], dim=1)
        return {
            'mean': self.decoder_net(z),
            'theta': torch.exp(self.theta_net(z)),
        }

    
class ParametricClusterer(nn.Module):
    def __init__(
        self, 
        n_clusters: int, 
        latent_dim: int,
        min_sigma: float = 0.00,
        max_sigma: float = 20.0,
    ):
        super().__init__()
        self.centroids = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.log_sigma = nn.Parameter(torch.zeros(n_clusters))
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.initialized = False

    
    def initialize_with_kmeans(self, z: torch.Tensor, target_similarity: float = 0.5):
        if self.initialized:
            return
            
        device = z.device
        self.to(device)
        
        from sklearn.cluster import KMeans
        z_np = z.detach().cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(z_np)

        with torch.no_grad():
            self.centroids.data = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float32)

            dist_sq = torch.sum(get_dist_sq(z, self.centroids), dim=-1)
            closest_centroids = dist_sq.argmin(dim=1)
            closest_centroids_mask = F.one_hot(closest_centroids, num_classes=self.n_clusters)
            avg_dist_sq = torch.sum(dist_sq * closest_centroids_mask, dim=0) / torch.sum(closest_centroids_mask, dim=0)
            self.log_sigma.data = (-avg_dist_sq / (2 * math.log(target_similarity))).clamp(min=self.min_sigma, max=self.max_sigma).log()

        self.initialized = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            return torch.ones(z.shape[0], self.n_clusters, device=z.device) / self.n_clusters

        centroids = self.centroids
        log_sigma = self.log_sigma

        sigma = torch.exp(log_sigma).clamp(min=self.min_sigma, max=self.max_sigma)
        
        return get_membership_scores(z, centroids, sigma)

def get_dist_sq(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    z_expanded = z.unsqueeze(1)  # (B, 1, D)
    c_expanded = centroids.unsqueeze(0)  # (1, K, D)
    dist_sq = 0.5 * (z_expanded - c_expanded).pow(2)  # (B, K, D)
    return dist_sq

def get_membership_scores(z: torch.Tensor, centroids: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_expanded = sigma.unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
    dist_sq = get_dist_sq(z, centroids) / sigma_expanded
    membership_logits = -torch.sum(dist_sq, dim=-1)  # (B, K)
    return F.softmax(membership_logits, dim=1)  # (B, K)
