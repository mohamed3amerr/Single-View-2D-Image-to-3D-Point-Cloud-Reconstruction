"""
Training script for the 3D Point Cloud VAE
"""
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from dataset import get_dataloaders
from models import PointCloudVAE, vae_loss


def train_vae(shapenet_root, num_models=3, num_epochs=50, batch_size=4, 
              lr=0.001, device='cuda', save_dir='checkpoints'):
    """
    Train the Point Cloud VAE.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        shapenet_root=shapenet_root,
        num_models=num_models,
        batch_size=batch_size,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = PointCloudVAE(latent_dim=256, num_points=2048).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        train_chamfer = []
        train_kl = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            points = batch['points'].to(device)
            
            # Forward pass
            recon, mu, logvar = model(points)
            loss, chamfer, kl = vae_loss(recon, points, mu, logvar, kl_weight=0.0001)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            train_losses.append(loss.item())
            train_chamfer.append(chamfer.item())
            train_kl.append(kl.item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'chamfer': f"{chamfer.item():.4f}",
                'kl': f"{kl.item():.4f}"
            })
        
        # Validation
        model.eval()
        val_losses = []
        val_chamfer = []
        val_kl = []
        
        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(device)
                
                recon, mu, logvar = model(points)
                loss, chamfer, kl = vae_loss(recon, points, mu, logvar, kl_weight=0.0001)
                
                val_losses.append(loss.item())
                val_chamfer.append(chamfer.item())
                val_kl.append(kl.item())
        
        # Scheduler step
        scheduler.step()
        
        # Print epoch summary
        train_loss_avg = np.mean(train_losses)
        val_loss_avg = np.mean(val_losses)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss_avg:.4f} | Chamfer: {np.mean(train_chamfer):.4f} | KL: {np.mean(train_kl):.4f}")
        print(f"Val Loss: {val_loss_avg:.4f} | Chamfer: {np.mean(val_chamfer):.4f} | KL: {np.mean(val_kl):.4f}")
        
        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(save_dir, 'vae_best.pth'))
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss_avg,
            }, os.path.join(save_dir, f'vae_epoch_{epoch+1}.pth'))
    
    print("Training completed!")
    return model


if __name__ == '__main__':
    # Configuration - Optimized for FAST 2-hour training
    SHAPENET_ROOT = '/Users/mohamedhady/Downloads/Deep learningv2/ShapeNetCore.v2/ShapeNetCore.v2'
    NUM_MODELS = 150  # 150 airplane models
    NUM_EPOCHS = 30  # 30 epochs for speed
    BATCH_SIZE = 16  # Larger batch for faster training
    LR = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    train_vae(
        shapenet_root=SHAPENET_ROOT,
        num_models=NUM_MODELS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE
    )

