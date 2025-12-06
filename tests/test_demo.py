import torch
import soundfile as sf
import numpy as np
import torchaudio.functional

from scripts.generate_final_dataset import TARGET_SR
from src.model.denoiseSystem import DenoiseSystem

BEST_MODEL_FILE_PATH = '../src/training/best_model.pt'

device = "cpu"
model = DenoiseSystem().to(device)
model.load_state_dict(torch.load(BEST_MODEL_FILE_PATH, map_location=device))
model.eval()

def demo_denoise_pipeline(result_path : str, noisy_file_path:str)-> None:
    audio, sr  = sf.read(noisy_file_path, dtype='float32')
    audio = torch.from_numpy(audio)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.mean(dim=1, keepdim=True).t()

    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)


    max_val  = audio.abs().max()
    if max_val > 1.0:
        audio = audio / max_val

    with torch.no_grad():
            noisy = audio.unsqueeze(0).to(device)
            denoised = model(noisy).squeeze().cpu().numpy()

    denoised_clipped = np.clip(denoised, -1.0, 1.0)

    sf.write(result_path, denoised_clipped, TARGET_SR)
    print("denoised audio created")



if __name__ == "__main__":
    demo_denoise_pipeline("./cleaned_files/cleand_directinput_01.wav", "../data/processed/directinput_01.wav")

