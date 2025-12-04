import os
import random

import torchaudio
from tqdm import tqdm
from pathlib import Path

from src.utils.audio_dataset import DenoisingDataSet
from src.utils.audio_processing_utils import *

RAW_PATH = Path('../data/raw')
TRAIN_PROCESSED_PATH = Path('../data/processed')
TRAIN_NOISE_PATH = Path('../data/noise')
TEST_CLEAN = Path('../data/test/clean')
TEST_NOISY = Path('../data/test/noisy')

TARGET_SR: int = int(os.getenv("TARGET_RESAMPLE_RATE", 16000))
SNR_RANGE: tuple[int, int] = (0, 20)
SEGMENT_LENGTH: int = TARGET_SR * 90


def process_dataset() -> None:
    EXTENSIONS = [".mp3", ".MP3", ".mp4", ".MP4", ".wav", ".WAV"]
    NOISE_EXTENSIONS = [".wav", ".WAV"]

    clean_files = [f for ext in EXTENSIONS for f in RAW_PATH.rglob(f"*{ext}")]
    noise_files = [f for ext in NOISE_EXTENSIONS for f in TRAIN_NOISE_PATH.rglob(f"*{ext}")]

    print("Noise and audio Files Loaded")

    if not clean_files:
        print("No clean audio files found.")
        return
    if not noise_files:
        print("No noise audio files found.")
        return

    TRAIN_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    for clean_file in tqdm(clean_files, desc="Generating noisy dataset"):
        try:
            create_noise_file(clean_file, noise_files)
        except Exception as e:
            print(f"Error processing {clean_file}: {e}")
            continue

def create_noise_file(clean_file, noise_files)->None:
        noise, clean = normalize_tensors(clean_file, noise_files)

        if noise.size(0) > 1:
            noise = noise.mean(dim=0, keepdim=True)

        snr = random.uniform(*SNR_RANGE)
        noisy = add_noise(clean, noise, snr)

        out_name = TRAIN_PROCESSED_PATH / f"{clean_file.stem}.wav"

        sf.write(str(out_name), noisy.squeeze(0).numpy(), TARGET_SR)




def split_training_data( size_training_dataset: int) -> None:
    dataset = DenoisingDataSet(noisy_dir=str(TRAIN_NOISE_PATH), clean_dir= str(TRAIN_PROCESSED_PATH))

    train_clean, train_noisy = dataset.clean_files, dataset.noisy_files

    counter = 0
    for clean_file, noisy_file in train_clean, train_noisy:
        while counter < size_training_dataset:
            os.replace(f"{TRAIN_PROCESSED_PATH}/{clean_file}" , f"{TEST_CLEAN}/{clean_file}")
            os.replace(f"{TRAIN_NOISE_PATH}/{noisy_file}" , f"{TEST_NOISY}/{noisy_file}")


#Important to note, to run this script you need a python envioerment with python version 3.12 to avoid dependency conflicts
#This script was run once already to generate our data set, only run if you wish to generate a new dataset, since noise addition function is randomized
if __name__ == "__main__":
    process_dataset()
    split_training_data(size_training_dataset=100)