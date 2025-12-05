"""
ShapeNet Dataset Loader for 2D to 3D reconstruction
"""
import os
import glob
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ShapeNetDataset(Dataset):
    """
    ShapeNet dataset that loads meshes, samples point clouds, and renders images.
    """
    
    def __init__(self, shapenet_root, num_models=None, num_points=2048, 
                 img_size=128, num_views=3, split='train', category='02691156'):
        """
        Args:
            shapenet_root: Path to ShapeNetCore.v2 directory
            num_models: Number of models to use (None = use all)
            num_points: Number of points to sample from each mesh
            img_size: Size of rendered images
            num_views: Number of views to render per model
            split: 'train' or 'test' (not used for now, just for future)
            category: ShapeNet category ID (default: 02691156 = airplanes)
        """
        self.shapenet_root = shapenet_root
        self.num_points = num_points
        self.img_size = img_size
        self.num_views = num_views
        
        # Find all model paths from specific category (airplanes)
        self.model_paths = []
        category_path = os.path.join(shapenet_root, category)
        
        if os.path.isdir(category_path):
            models = sorted(os.listdir(category_path))
            for model_id in models:
                model_path = os.path.join(category_path, model_id, 'models', 'model_normalized.ply')
                if os.path.exists(model_path):
                    self.model_paths.append(model_path)
                    if num_models is not None and len(self.model_paths) >= num_models:
                        break
        
        print(f"Found {len(self.model_paths)} models")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.model_paths) * self.num_views
    
    def load_mesh_and_sample_points(self, mesh_path):
        """Load mesh and sample points uniformly from surface"""
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # Sample points from mesh surface
        pcd = mesh.sample_points_uniformly(number_of_points=self.num_points)
        points = np.asarray(pcd.points)
        
        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / (max_dist + 1e-8)
        
        return points.astype(np.float32)
    
    def render_image(self, mesh_path, view_idx):
        """
        Render an image from a mesh at a specific viewpoint.
        Uses simple matplotlib-based rendering to avoid Open3D visualization issues.
        """
        # Load mesh and convert to point cloud for simple visualization
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        pcd = mesh.sample_points_uniformly(number_of_points=4096)
        points = np.asarray(pcd.points)
        
        # Normalize points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / (max_dist + 1e-8)
        
        # Different viewpoints - rotate points
        angles = [0, 90, 45]  # rotation angles around Y axis
        angle = angles[view_idx % len(angles)]
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        points = points @ rotation_matrix.T
        
        # Project to 2D (simple orthographic projection)
        # Use matplotlib to render
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig = plt.figure(figsize=(self.img_size/100, self.img_size/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        ax.view_init(elev=20, azim=angle)
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array)
        
        return img
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Rendered 2D image (3, H, W)
            points: Ground truth point cloud (N, 3)
        """
        model_idx = idx // self.num_views
        view_idx = idx % self.num_views
        
        mesh_path = self.model_paths[model_idx]
        
        # Load point cloud
        points = self.load_mesh_and_sample_points(mesh_path)
        
        # Render image
        img = self.render_image(mesh_path, view_idx)
        img = self.transform(img)
        
        return {
            'image': img,
            'points': torch.from_numpy(points),
            'model_path': mesh_path
        }


def get_dataloaders(shapenet_root, num_models=None, batch_size=4, num_workers=0):
    """
    Create train and validation dataloaders.
    For simplicity, we use the same data for both.
    """
    dataset = ShapeNetDataset(
        shapenet_root=shapenet_root,
        num_models=num_models,
        num_points=2048,
        img_size=128,
        num_views=3,
        category='02691156'  # Airplanes only
    )
    
    # Simple split: use 80% for training, 20% for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader

