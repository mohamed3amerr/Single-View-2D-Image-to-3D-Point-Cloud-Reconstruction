"""
Model architectures for 3D VAE and Image Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for point clouds.
    Encodes a point cloud into a latent vector.
    """
    
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Point-wise MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global feature to latent
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)
        
        # Point-wise features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, 256)
        
        # Latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class PointCloudDecoder(nn.Module):
    """
    MLP decoder that generates a point cloud from a latent vector.
    """
    
    def __init__(self, latent_dim=256, num_points=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, num_points * 3)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)
        Returns:
            points: (B, num_points, 3)
        """
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        
        # Reshape to point cloud
        points = x.view(-1, self.num_points, 3)
        
        # Apply tanh to keep points in reasonable range
        points = torch.tanh(points)
        
        return points


class PointCloudVAE(nn.Module):
    """
    Variational Autoencoder for point clouds.
    """
    
    def __init__(self, latent_dim=256, num_points=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = PointCloudDecoder(latent_dim, num_points)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) input point cloud
        Returns:
            recon: (B, N, 3) reconstructed point cloud
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x):
        """Encode point cloud to latent vector (using mean)"""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        """Decode latent vector to point cloud"""
        return self.decoder(z)


class ImageEncoder(nn.Module):
    """
    ResNet18-based image encoder that maps 2D images to the same latent space as the VAE.
    """
    
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Use pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom FC layer to project to latent space
        self.fc = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image
        Returns:
            z: (B, latent_dim) latent vector
        """
        # Extract features
        features = self.features(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        
        # Project to latent space
        z = self.fc(features)
        
        return z


def chamfer_distance(pred, target):
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pred: (B, N, 3) predicted point cloud
        target: (B, M, 3) target point cloud
    Returns:
        loss: scalar Chamfer distance
    """
    # pred: (B, N, 3), target: (B, M, 3)
    
    # Compute pairwise distances
    # pred: (B, N, 1, 3), target: (B, 1, M, 3)
    pred_expand = pred.unsqueeze(2)  # (B, N, 1, 3)
    target_expand = target.unsqueeze(1)  # (B, 1, M, 3)
    
    # Squared distances: (B, N, M)
    dist = torch.sum((pred_expand - target_expand) ** 2, dim=3)
    
    # Chamfer distance
    dist_pred_to_target = torch.min(dist, dim=2)[0]  # (B, N)
    dist_target_to_pred = torch.min(dist, dim=1)[0]  # (B, M)
    
    chamfer_dist = torch.mean(dist_pred_to_target) + torch.mean(dist_target_to_pred)
    
    return chamfer_dist


def vae_loss(recon, target, mu, logvar, kl_weight=0.0001):
    """
    VAE loss = Chamfer distance + KL divergence.
    
    Args:
        recon: (B, N, 3) reconstructed point cloud
        target: (B, N, 3) target point cloud
        mu: (B, latent_dim)
        logvar: (B, latent_dim)
        kl_weight: weight for KL divergence term
    Returns:
        loss: scalar loss
        chamfer: scalar Chamfer distance
        kl: scalar KL divergence
    """
    # Chamfer distance
    chamfer = chamfer_distance(recon, target)
    
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # Total loss
    loss = chamfer + kl_weight * kl
    
    return loss, chamfer, kl

