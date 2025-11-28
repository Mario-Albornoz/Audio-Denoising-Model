import os
import soundfile as sf


def check_audio_files(directory):
    corrupted = []
    valid = []

    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            try:
                # Try to read file info
                info = sf.info(filepath)
                valid.append((filename, info.samplerate, info.duration))
                print(f"✓ {filename}: {info.duration:.2f}s, {info.samplerate}Hz")
            except Exception as e:
                corrupted.append((filename, str(e)))
                print(f"✗ {filename}: {e}")

                # Check file size
                size = os.path.getsize(filepath)
                print(f"  File size: {size} bytes")

    print(f"\n{len(valid)} valid files, {len(corrupted)} corrupted files")
    return corrupted, valid


# Check both directories
print("Checking noisy files:")
corrupted_noisy, _ = check_audio_files("../data/processed")

print("\nChecking clean files:")
corrupted_clean, _ = check_audio_files("../data/raw")