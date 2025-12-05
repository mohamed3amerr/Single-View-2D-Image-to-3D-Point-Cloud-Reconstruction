"""
Complete training script for better results - trains on more objects
"""
import os
import sys

# Train VAE on more models
print("="*60)
print("STEP 1: Training VAE on 20 airplane models")
print("="*60)
os.system("export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && python3 train_vae.py")

print("\n" + "="*60)
print("STEP 2: Training Image Encoder")
print("="*60)
os.system("export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && python3 train_image_encoder.py")

print("\n" + "="*60)
print(" Training Complete!")
print("="*60)
print("Run the app: python3 -m streamlit run app.py")

