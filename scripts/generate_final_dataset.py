import os
import random

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path

RAW_PATH = Path('../data/raw')
PROCESSED_PATH = Path('../data/processed')
NOISE_PATH = Path('../data/noise')

TARGET_SR: int = int(os.getenv("TARGET_RESAMPLE_RATE", 16000))
SNR_RANGE: tuple[int, int] = (0, 20)
SEGMENT_LENGTH: int = TARGET_SR * 90


def load_audio_soundfile(file_path):
    """Load audio using soundfile directly"""
    data, sr = sf.read(str(file_path), dtype='float32')
    waveform = torch.from_numpy(data.T)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, sr


def add_noise(clean, noise, snr_db):
    if noise.size(1) < clean.size(1):
        repeats = (clean.size(1) // noise.size(1)) + 1
        noise = noise.repeat(1, repeats)
    noise = noise[:, :clean.size(1)]

    snr = 10 ** (snr_db / 10)
    clean_power = clean.norm(p=2)
    noise_power = noise.norm(p=2)
    scale = clean_power / (snr * noise_power)

    return clean + scale * noise


def process_dataset() -> None:
    EXTENSIONS = [".mp3", ".MP3", ".mp4", ".MP4", ".wav", ".WAV"]
    NOISE_EXTENSIONS = [".wav", ".WAV"]

    clean_files = [f for ext in EXTENSIONS for f in RAW_PATH.rglob(f"*{ext}")]
    noise_files = [f for ext in NOISE_EXTENSIONS for f in NOISE_PATH.rglob(f"*{ext}")]

    print("Noise and audio Files Loaded")

    if not clean_files:
        print("No clean audio files found.")
        return
    if not noise_files:
        print("No noise audio files found.")
        return

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    for clean_file in tqdm(clean_files, desc="Generating noisy dataset"):
        try:
            clean, sr = load_audio_soundfile(clean_file)

            if sr != TARGET_SR:
                clean = torchaudio.functional.resample(clean, sr, TARGET_SR)

            if clean.size(0) > 1:
                clean = clean.mean(dim=0, keepdim=True)

            if clean.size(1) > SEGMENT_LENGTH:
                start = random.randint(0, clean.size(1) - SEGMENT_LENGTH)
                clean = clean[:, start:start + SEGMENT_LENGTH]

            noise_file = random.choice(noise_files)
            noise, nsr = load_audio_soundfile(noise_file)

            if nsr != TARGET_SR:
                noise = torchaudio.functional.resample(noise, nsr, TARGET_SR)

            if noise.size(0) > 1:
                noise = noise.mean(dim=0, keepdim=True)

            snr = random.uniform(*SNR_RANGE)
            noisy = add_noise(clean, noise, snr)

            out_name = PROCESSED_PATH / f"{clean_file.stem}.wav"

            sf.write(str(out_name), noisy.squeeze(0).numpy(), TARGET_SR)

        except Exception as e:
            print(f"Error processing {clean_file}: {e}")
            continue


#Important to note, to run this script you need a python envioerment with python version 3.12 to avoid dependency conflicts
#This script was run once already to generate our data set, only run if you wish to generate a new dataset, since noise addition function is randomized
#TODO:multithread this script to increase the completion time
if __name__ == "__main__":
    process_dataset()