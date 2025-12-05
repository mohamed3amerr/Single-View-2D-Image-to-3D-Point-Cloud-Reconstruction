"""
Streamlit GUI for 2D Image to 3D Point Cloud Reconstruction
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import torchvision.transforms as transforms

from models import PointCloudVAE, ImageEncoder


@st.cache_resource
def load_models(vae_path, encoder_path, device='cpu'):
    """Load the trained models"""
    # Load VAE
    vae = PointCloudVAE(latent_dim=256, num_points=2048).to(device)
    vae_checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    
    # Load Image Encoder
    image_encoder = ImageEncoder(latent_dim=256).to(device)
    encoder_checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    image_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    image_encoder.eval()
    
    return vae, image_encoder


def preprocess_image(image, img_size=128):
    """Preprocess uploaded image"""
    # Resize and convert to RGB
    image = image.convert('RGB')
    image = image.resize((img_size, img_size))
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict_3d(image, vae, image_encoder, device='cpu'):
    """Predict 3D point cloud from 2D image"""
    with torch.no_grad():
        # Encode image to latent
        z = image_encoder(image.to(device))
        
        # Decode to point cloud
        points = vae.decode(z)
        
        # Convert to numpy
        points = points.cpu().numpy()[0]  # (N, 3)
    
    return points


def visualize_point_cloud(points):
    """Create interactive 3D visualization with Plotly"""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],  # Color by Z coordinate
            colorscale='Viridis',
            showscale=True
        )
    )])
    
    fig.update_layout(
        title="Predicted 3D Point Cloud",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'
        ),
        width=700,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig


def main():
    st.set_page_config(page_title="2D to 3D Reconstruction", layout="wide")
    
    st.title(" 2D Image to 3D Point Cloud Reconstruction")
    st.markdown("Upload a 2D image of an object and see its predicted 3D shape!")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"Using device: **{device}**")
    
    vae_path = st.sidebar.text_input("VAE Checkpoint Path", "checkpoints/vae_best.pth")
    encoder_path = st.sidebar.text_input("Encoder Checkpoint Path", "checkpoints/image_encoder_best.pth")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("Predicted 3D Shape")
        
        if uploaded_file is not None:
            if st.button("Generate 3D Model", type="primary"):
                with st.spinner("Loading models..."):
                    try:
                        vae, image_encoder = load_models(vae_path, encoder_path, device)
                        st.success("Models loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading models: {e}")
                        return
                
                with st.spinner("Generating 3D point cloud..."):
                    try:
                        # Preprocess image
                        image_tensor = preprocess_image(image)
                        
                        # Predict 3D
                        points = predict_3d(image_tensor, vae, image_encoder, device)
                        
                        # Visualize
                        fig = visualize_point_cloud(points)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show statistics
                        st.success(f"Generated {len(points)} points!")
                        st.info(f"Point cloud bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                               f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                               f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
                        
                        # Download option
                        points_str = '\n'.join([f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}" for p in points])
                        st.download_button(
                            label="Download Point Cloud (CSV)",
                            data=points_str,
                            file_name="point_cloud.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
        else:
            st.info(" Upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About
    This application uses a **3D VAE** and **Image Encoder** to reconstruct 3D point clouds from 2D images.
    
    **Architecture:**
    - **3D VAE**: PointNet encoder + MLP decoder
    - **Image Encoder**: ResNet18-based CNN
    - **Training**: Chamfer distance + KL divergence + Latent space alignment
    """)


if __name__ == '__main__':
    main()

