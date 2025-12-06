import torch
from src.utils.audio_dataset import DenoisingDataSet
import soundfile as sf

print("="*60)
print("CHECKING TRAINING DATA FORMAT")
print("="*60)

# Load dataset the same way training does
dataset = DenoisingDataSet(
    noisy_dir="../data/processed",
    clean_dir="../data/raw",
    target_sr=16000,
    segment_length=80000
)

print(f"\nDataset has {len(dataset)} samples")

# Get first sample
sample = dataset[0]
print(f"\nSample 0: {sample['stem']}")

noisy = sample['noisy']
clean = sample['clean']

print(f"\nNoisy tensor:")
print(f"  Shape: {noisy.shape}")
print(f"  Range: [{noisy.min():.4f}, {noisy.max():.4f}]")
print(f"  Mean: {noisy.mean():.4f}")

print(f"\nClean tensor:")
print(f"  Shape: {clean.shape}")
print(f"  Range: [{clean.min():.4f}, {clean.max():.4f}]")
print(f"  Mean: {clean.mean():.4f}")

# Calculate difference
diff = (noisy - clean).abs().mean()
print(f"\nDifference (noisy - clean): {diff:.6f}")

# Calculate SNR
noise = noisy - clean
signal_power = (clean ** 2).sum()
noise_power = (noise ** 2).sum()

if noise_power > 0:
    snr = 10 * torch.log10(signal_power / noise_power)
    print(f"SNR: {snr:.2f} dB")
else:
    print("SNR: Infinite (no noise!)")

# Save samples to listen
print(f"\nSaving samples for inspection...")
sf.write("training_noisy_sample.wav", noisy[0].numpy(), 16000)
sf.write("training_clean_sample.wav", clean[0].numpy(), 16000)

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if diff < 0.001:
    print("❌ CRITICAL: Noisy and clean are almost identical!")
    print("   Your training data has no noise.")
    print("   The model learned nothing useful.")
elif diff < 0.01:
    print("⚠️  WARNING: Very little noise in training data")
    print("   Model might not have learned to denoise properly")
else:
    print("✓ Training data looks reasonable")
    print("  But check why model performs poorly...")

print("\nAction items:")
print("1. Listen to training_noisy_sample.wav")
print("2. Listen to training_clean_sample.wav")
print("3. Are they clearly different?")