import torch
import soundfile as sf
import numpy as np
import torchaudio.functional as F
from pathlib import Path

from scripts.generate_final_dataset import TARGET_SR
from src.model.demucs import BaseDemucs

BEST_MODEL_FILE_PATH = '../src/training/best_model.pt'

device = "cpu"
model = BaseDemucs().to(device)
model.load_state_dict(torch.load(BEST_MODEL_FILE_PATH, map_location=device))
model.eval()


def rescue_broken_model_output(denoised, original_noisy, alpha=0.3):

    denoised_norm = denoised / (denoised.abs().max() + 1e-8)
    noisy_norm = original_noisy / (original_noisy.abs().max() + 1e-8)

    mixed = (1 - alpha) * denoised_norm + alpha * noisy_norm

    mixed = mixed * original_noisy.abs().max()

    return mixed

def demo_denoise_pipeline(result_path: str, noisy_file_path: str) -> None:

    audio, sr = sf.read(noisy_file_path, dtype='float32')
    print(f"Loaded audio: shape={audio.shape}, sr={sr}")

    audio = torch.from_numpy(audio)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        if audio.shape[0] > audio.shape[1]:
            audio = audio.t()
        audio = audio.mean(dim=0, keepdim=True)

    print(f"After mono conversion: {audio.shape}")

    if sr != TARGET_SR:
        print(f"Resampling from {sr}Hz to {TARGET_SR}Hz")
        audio = F.resample(audio, sr, TARGET_SR)

    max_val = audio.abs().max()
    if max_val > 1.0:
        print(f"Normalizing audio (max value: {max_val:.3f})")
        audio = audio / max_val

    audio = audio.unsqueeze(0)
    print(f"Model input shape: {audio.shape}")

    with torch.no_grad():
        noisy = audio.to(device)
        denoised = model(noisy)

        denoised = denoised.squeeze().cpu().numpy()

    print(f"Denoised output shape: {denoised.shape}")

    denoised_clipped = np.clip(denoised, -1.0, 1.0)

    Path(result_path).parent.mkdir(parents=True, exist_ok=True)

    sf.write(result_path, denoised_clipped, TARGET_SR)

    print(f"\nStatistics:")
    print(f"  Input max amplitude: {np.abs(audio.squeeze().numpy()).max():.3f}")
    print(f"  Output max amplitude: {np.abs(denoised_clipped).max():.3f}")
    print(f"  Output length: {len(denoised_clipped) / TARGET_SR:.2f} seconds")


if __name__ == "__main__":
    demo_denoise_pipeline("./cleaned_files/cleand_directinput_01.wav", "../data/processed/exo_Set3_aug.wav")

