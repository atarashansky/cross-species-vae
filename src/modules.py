from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedded = self.embed(x)
        return self.encode(embedded)

class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        hidden_dims: list,
        n_clusters: int,
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
                
        dims = [n_latent + n_clusters] + hidden_dims + [n_genes]
        
        self.decoder_net = build_decoder_layers(dims)
        self.theta_net = build_decoder_layers(dims)


    
    def forward(self, z: torch.Tensor, memberships: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_cat = torch.cat([z, memberships], dim=-1)
        return {
            'mean': self.decoder_net(z_cat),
            'theta': torch.exp(self.theta_net(z_cat))
        }

class ParametricClusterer(nn.Module):
    def __init__(
        self, 
        n_clusters: int, 
        latent_dim: int,
        initial_sigma: float = 3.0,
        min_sigma: float = 0.1,
        max_sigma: float = 10.0,
    ):
        super().__init__()
        self.centroids = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.log_sigma = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.initial_sigma = initial_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.initialized = False


    def initialize_with_kmeans(self, z: torch.Tensor):
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
            self.log_sigma.data = torch.log(torch.full((self.n_clusters, self.latent_dim), self.initial_sigma, device=device, dtype=torch.float32)) 
            
        self.initialized = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            return torch.ones((z.shape[0], self.n_clusters), device=z.device) / self.n_clusters
        
        centroids = self.centroids
        log_sigma = self.log_sigma

        sigma = torch.exp(log_sigma).clamp(min=self.min_sigma, max=self.max_sigma)
        
        z_expanded = z.unsqueeze(1)  # (B, 1, D)
        c_expanded = centroids.unsqueeze(0)  # (1, K, D)
        dist_sq = torch.sum((z_expanded - c_expanded)**2, dim=-1)  # (B, K)
        
        membership_logits = -dist_sq / (2 * sigma)
        memberships = F.softmax(membership_logits, dim=1)  # (B, K)
        
        return memberships


class VaDEClusterer(nn.Module):
    def __init__(
        self, 
        latent_dim: int,
        n_clusters: int,
        initial_sigma: float = 2.0,
        min_sigma: float = 0.00,
        max_sigma: float = 20.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.initial_sigma = initial_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # GMM parameters
        self.centroids = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.log_sigma = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.pi_logits = nn.Parameter(torch.log(torch.ones(n_clusters) / n_clusters))
        self.initialized = False

    def initialize_with_kmeans(self, z: torch.Tensor):
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
            self.log_sigma.data = torch.log(torch.full((self.n_clusters,self.latent_dim), self.initial_sigma, device=device, dtype=torch.float32)) 
            
        self.initialized = True
        

    def initialize_with_gmm(self, z: torch.Tensor):
        if self.initialized:
            return
            
        device = z.device
        self.to(device)
        
        from sklearn.mixture import GaussianMixture
        z_np = z.detach().cpu().numpy()
        
        gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='diag',
            max_iter=100,
            random_state=42
        )
        gmm.fit(z_np)

        with torch.no_grad():
            self.centroids.data = torch.tensor(gmm.means_, device=device, dtype=torch.float32)
            self.log_sigma.data = torch.log(torch.tensor(gmm.covariances_, device=device, dtype=torch.float32)) 
            
        self.initialized = True

    def get_gamma(self, z: torch.Tensor):
        pi_tensor = torch.tensor(torch.pi, device=z.device)
        Z = z.unsqueeze(1)
        centroids = self.centroids.unsqueeze(0)
        log_sigma = self.log_sigma.unsqueeze(0)
        pi_logits = self.pi_logits.unsqueeze(0)

        log_p_c_z = (
            pi_logits - 
            torch.sum(
                0.5 * torch.log( 2 * pi_tensor) + log_sigma + (Z-centroids).pow(2) / (2 * torch.exp(log_sigma)),
                dim=-1
            )
        )
        gamma = torch.exp(log_p_c_z - torch.logsumexp(log_p_c_z, dim=1, keepdim=True))
        return gamma
    
    def forward(self, encoded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_mean = encoded['mu'].clamp(min=-20, max=20)
        z_logvar = encoded['logvar'].clamp(min=-20, max=20)
        z = encoded['z'].clamp(min=-20, max=20)

        if not self.initialized:
            uniform = torch.ones((z_mean.shape[0], self.n_clusters), device=z_mean.device) / self.n_clusters
            return {"memberships": uniform, "gmm_loss": torch.tensor(0.0, device=z_mean.device)}
        
        pi_tensor = torch.tensor(torch.pi, device=z_mean.device)
        sigma = torch.exp(self.log_sigma).clamp(min=self.min_sigma, max=self.max_sigma)
        gamma = self.get_gamma(z)

        logpzc = torch.sum(0.5*gamma*torch.sum(torch.log(2*pi_tensor)+self.log_sigma+torch.exp(z_logvar.unsqueeze(1))/sigma.unsqueeze(0) + (z_mean.unsqueeze(1)-self.centroids.unsqueeze(0)).pow(2)/sigma.unsqueeze(0), dim=-1), dim=-1)
        
        qentropy = -0.5*torch.sum(1+z_logvar+torch.log(2*pi_tensor), 1)
        logpc = -torch.sum(self.pi_logits * gamma, 1)
        logqcx = torch.sum(torch.log(gamma + 1e-10)*gamma, 1)        

        gmm_loss = (logpzc + qentropy + logpc + logqcx).mean()
        
        return {
            "memberships": gamma,
            "gmm_loss": gmm_loss
        }