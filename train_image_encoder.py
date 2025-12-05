"""
Training script for the Image Encoder
"""
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from dataset import get_dataloaders
from models import PointCloudVAE, ImageEncoder, chamfer_distance


def train_image_encoder(shapenet_root, vae_checkpoint, num_models=3, num_epochs=50, 
                       batch_size=4, lr=0.001, device='cuda', save_dir='checkpoints'):
    """
    Train the Image Encoder to map images to the VAE latent space.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load VAE (frozen)
    print("Loading VAE...")
    vae = PointCloudVAE(latent_dim=256, num_points=2048).to(device)
    checkpoint = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        shapenet_root=shapenet_root,
        num_models=num_models,
        batch_size=batch_size,
        num_workers=0
    )
    
    # Create image encoder
    print("Creating image encoder...")
    image_encoder = ImageEncoder(latent_dim=256).to(device)
    
    # Optimizer
    optimizer = optim.Adam(image_encoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        image_encoder.train()
        train_losses = []
        train_latent_losses = []
        train_chamfer_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            points = batch['points'].to(device)
            
            # Get ground truth latent from VAE
            with torch.no_grad():
                z_gt = vae.encode(points)
            
            # Forward pass through image encoder
            z_img = image_encoder(images)
            
            # Decode to point cloud
            pred_points = vae.decode(z_img)
            
            # Loss: MSE in latent space + Chamfer distance
            latent_loss = torch.nn.functional.mse_loss(z_img, z_gt)
            chamfer_loss = chamfer_distance(pred_points, points)
            
            loss = latent_loss + 0.1 * chamfer_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            train_losses.append(loss.item())
            train_latent_losses.append(latent_loss.item())
            train_chamfer_losses.append(chamfer_loss.item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'latent': f"{latent_loss.item():.4f}",
                'chamfer': f"{chamfer_loss.item():.4f}"
            })
        
        # Validation
        image_encoder.eval()
        val_losses = []
        val_latent_losses = []
        val_chamfer_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                points = batch['points'].to(device)
                
                z_gt = vae.encode(points)
                z_img = image_encoder(images)
                pred_points = vae.decode(z_img)
                
                latent_loss = torch.nn.functional.mse_loss(z_img, z_gt)
                chamfer_loss = chamfer_distance(pred_points, points)
                loss = latent_loss + 0.1 * chamfer_loss
                
                val_losses.append(loss.item())
                val_latent_losses.append(latent_loss.item())
                val_chamfer_losses.append(chamfer_loss.item())
        
        # Scheduler step
        scheduler.step()
        
        # Print epoch summary
        train_loss_avg = np.mean(train_losses)
        val_loss_avg = np.mean(val_losses)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss_avg:.4f} | Latent: {np.mean(train_latent_losses):.4f} | Chamfer: {np.mean(train_chamfer_losses):.4f}")
        print(f"Val Loss: {val_loss_avg:.4f} | Latent: {np.mean(val_latent_losses):.4f} | Chamfer: {np.mean(val_chamfer_losses):.4f}")
        
        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': image_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(save_dir, 'image_encoder_best.pth'))
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': image_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss_avg,
            }, os.path.join(save_dir, f'image_encoder_epoch_{epoch+1}.pth'))
    
    print("Training completed!")
    return image_encoder


if __name__ == '__main__':
    # Configuration - Optimized for FAST 2-hour training
    SHAPENET_ROOT = '/Users/mohamedhady/Downloads/Deep learningv2/ShapeNetCore.v2/ShapeNetCore.v2'
    VAE_CHECKPOINT = 'checkpoints/vae_best.pth'
    NUM_MODELS = 150  # 150 airplane models
    NUM_EPOCHS = 30  # 30 epochs for speed
    BATCH_SIZE = 16  # Larger batch for faster training
    LR = 0.0001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    train_image_encoder(
        shapenet_root=SHAPENET_ROOT,
        vae_checkpoint=VAE_CHECKPOINT,
        num_models=NUM_MODELS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE
    )

