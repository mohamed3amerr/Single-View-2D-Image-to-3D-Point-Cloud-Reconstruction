"""
Fast training script - completes in ~2 hours
Trains on 200 airplane models with 50 epochs each
"""
import subprocess
import time

print("="*70)
print(" FAST TRAINING MODE - Complete in ~2 hours")
print("="*70)
print("\n Configuration:")
print("   - Models: 150 airplanes")
print("   - Epochs: 30 per model")
print("   - Batch size: 16")
print("   - Total time: ~2 hours")
print("\n" + "="*70)

# Train VAE
print("\n STEP 1/2: Training 3D VAE (~1 hour)")
print("="*70)
start_vae = time.time()
result = subprocess.run(
    "export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && python3 train_vae.py",
    shell=True,
    cwd="/Users/mohamedhady/Downloads/Deep learningv2"
)
vae_time = (time.time() - start_vae) / 60

if result.returncode != 0:
    print(f"\n VAE training failed!")
    exit(1)

print(f"\n VAE training completed in {vae_time:.1f} minutes")

# Train Image Encoder
print("\n STEP 2/2: Training Image Encoder (~1 hour)")
print("="*70)
start_encoder = time.time()
result = subprocess.run(
    "export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && python3 train_image_encoder.py",
    shell=True,
    cwd="/Users/mohamedhady/Downloads/Deep learningv2"
)
encoder_time = (time.time() - start_encoder) / 60

if result.returncode != 0:
    print(f"\n Image encoder training failed!")
    exit(1)

print(f"\n Image encoder training completed in {encoder_time:.1f} minutes")

total_time = vae_time + encoder_time
print("\n" + "="*70)
print(" ALL TRAINING COMPLETE!")
print("="*70)
print(f"   VAE training:      {vae_time:.1f} minutes")
print(f"   Encoder training:  {encoder_time:.1f} minutes")
print(f"   Total time:        {total_time:.1f} minutes ({total_time/60:.1f} hours)")
print("\n Launch the app: python3 -m streamlit run app.py")
print("="*70)

