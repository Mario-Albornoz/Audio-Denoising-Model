import torch
import soundfile as sf
import torchaudio.functional as F
from pathlib import Path

from scripts.generate_final_dataset import TARGET_SR
from src.evaluation.metrics import MultiScaleLoss, compute_snr
from src.model.demucs import BaseDemucs
from src.utils.audio_processing_utils import find_corrupted_files

BEST_MODEL_FILE_PATH = '../training/best_model.pt'
TRAIN_CLEAN_DIR_PATH = "../../data/train/clean"
TRAIN_NOISY_DIR_PATH = "../../data/train/noisy"
TEST_NOISY_DIR_PATH = "../../data/test/noisy"
TEST_CLEAN_DIR_PATH = "../../data/test/clean"

def clean_corrupted_files(noisy_dir, clean_dir) -> tuple[list, list]:
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)
    unclean_noise_files = list(noisy_dir.glob("*.wav"))
    unclean_raw_files = list(clean_dir.glob("*.wav"))

    corrupted_clean_files = find_corrupted_files(clean_dir)
    corrupted_noisy_files = find_corrupted_files(noisy_dir)
    joined_corrupted_files = set(corrupted_noisy_files + corrupted_clean_files)

    noise_files = sorted([f for f in unclean_noise_files if f not in joined_corrupted_files])
    raw_files = sorted([f for f in unclean_raw_files if f not in joined_corrupted_files])

    return noise_files, raw_files

def validate_and_normalize_audio(audio, sr):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        if audio.shape[0] > audio.shape[1]:
            audio = audio.t()
        audio = audio.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        audio = F.resample(audio, sr, TARGET_SR)

    max_val = audio.abs().max()
    if max_val > 1.0:
        audio = audio / max_val

    audio = audio.unsqueeze(0)

    return audio

def test_model(noisy_dir_path, clean_dir_path):
    print("starting program")
    device = "cpu"
    model = BaseDemucs().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_FILE_PATH, map_location=device))
    model.eval()
    print('loaded model')
    noisy_files, clean_files = clean_corrupted_files(noisy_dir_path, clean_dir_path)
    print("loaded clen and noise file")

    avg_loss = 0
    avg_time_loss = 0
    avg_spectral_loss = 0
    avg_snr = 0
    loss_function = MultiScaleLoss()

    print('starting training loop...')
    for idx, (noisy_file, clean_file) in enumerate(zip(noisy_files, clean_files), 1):
        try:
            noisy_audio, sr = sf.read(noisy_file, dtype='float32')
            clean_audio, clean_sr = sf.read(clean_file, dtype='float32')
        except Exception as e:
            print(f"Error trying to open {noisy_file} : {e}, skipping...")
            continue


        noisy_audio = torch.from_numpy(noisy_audio)
        noisy_audio = validate_and_normalize_audio(noisy_audio, sr)

        with torch.no_grad():
            enhanced_audio = model(noisy_audio)

        clean_audio = torch.from_numpy(clean_audio)
        clean_audio = validate_and_normalize_audio(clean_audio, clean_sr)

        loss, time_loss, spectral_loss = loss_function(enhanced_audio, clean_audio)
        snr = compute_snr(clean=clean_audio, enhanced=enhanced_audio)

        avg_loss += loss.item()
        avg_time_loss += time_loss.item()
        avg_spectral_loss += spectral_loss.item()
        avg_snr += snr.item()

        del clean_audio, noisy_audio, loss, spectral_loss, time_loss


    num_samples = len(noisy_files)
    avg_loss /= num_samples
    avg_time_loss /= num_samples
    avg_spectral_loss /= num_samples
    avg_snr /= num_samples

    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal samples processed: {num_samples}")
    print(f"\nAverage Total Loss:     {avg_loss:.6f}")
    print(f"Average Time Loss:      {avg_time_loss:.6f}")
    print(f"Average Spectral Loss:  {avg_spectral_loss:.6f}")
    print(f"Average snr :  {avg_snr:.6f}")
    print("\n" + "=" * 60)

    return {
        'avg_loss': avg_loss,
        'avg_time_loss': avg_time_loss,
        'avg_spectral_loss': avg_spectral_loss,
        'num_samples': num_samples
    }

if __name__ == "__main__":
    test_model(noisy_dir_path=TRAIN_NOISY_DIR_PATH,clean_dir_path=TRAIN_CLEAN_DIR_PATH)
    test_model(noisy_dir_path=TEST_NOISY_DIR_PATH, clean_dir_path=TEST_CLEAN_DIR_PATH)