import os

from tqdm import tqdm
from pathlib import Path

from src.utils.audio_processing_utils import *

RAW_PATH = Path('../data/raw')
NOISE_PATH = Path('../data/noise')
TRAIN_CLEAN_PATH = Path('../data/train/clean')
TRAIN_NOISY_PATH = Path('../data/train/noisy')
TEST_CLEAN_PATH = Path('../data/test/clean')
TEST_NOISY_PATH = Path('../data/test/noisy')

TARGET_SR: int = 44100
SNR_RANGE: tuple[int, int] = (5, 15)
SEGMENT_LENGTH: int = TARGET_SR * 90

def generate_dataset():
    EXTENSIONS = [".mp3", ".MP3", ".mp4", ".MP4", ".wav", ".WAV"]
    NOISE_EXTENSIONS = [".wav", ".WAV"]

    clean_files = [f for ext in EXTENSIONS for f in RAW_PATH.rglob(f"*{ext}")]
    pure_noise_files = [f for ext in NOISE_EXTENSIONS for f in NOISE_PATH.rglob(f"*{ext}")]

    if not clean_files:
        raise ValueError(f"{len(clean_files)} files in clean files, check path {str(RAW_PATH)}")
    if not pure_noise_files:
        raise ValueError(f"{len(pure_noise_files)} files in clean files, check path {str(NOISE_PATH)}")

    TRAIN_NOISY_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_CLEAN_PATH.mkdir(parents=True, exist_ok=True)
    TEST_CLEAN_PATH.mkdir(parents=True, exist_ok=True)
    TEST_NOISY_PATH.mkdir(parents=True, exist_ok=True)

    for clean_file  in tqdm(clean_files, desc='Generating noisy dataset'):
        try:
            create_noise_file(clean_file, pure_noise_file=pure_noise_files)
        except Exception as e:
            print(f'Error processing: {e}')
            continue

def create_noise_file(clean_file, pure_noise_file)->None:
        noise, clean = normalize_tensors(str(clean_file), pure_noise_file, TARGET_SR, SEGMENT_LENGTH)

        if noise.size(0) > 1:
            noise = noise.mean(dim=0, keepdim=True)

        snr = random.uniform(*SNR_RANGE)
        noisy = add_noise(clean, noise, snr)

        clean_file_out_name = TRAIN_CLEAN_PATH / f"{clean_file.stem}.wav"
        noisy_file_out_name = TRAIN_NOISY_PATH/ f"{clean_file.stem}.wav"

        sf.write(str(clean_file_out_name), clean.squeeze(0).numpy(), TARGET_SR)
        sf.write(str(noisy_file_out_name), noisy.squeeze(0).numpy(), TARGET_SR)

def split_dataset(size_training_dataset: int) -> None:
    train_clean_files = sorted(TRAIN_CLEAN_PATH.glob("*.wav"))
    train_noisy_files = sorted(TRAIN_NOISY_PATH.glob("*.wav"))

    if len(train_noisy_files) != len(train_clean_files):
        raise ValueError(f'Could not split data set '
                         f'Length train noisy file: {len(train_noisy_files)} and length train clean files: {len(train_clean_files)} do not match')

    max_train_size = len(train_clean_files) * 9 // 10
    train_size = min(max_train_size, size_training_dataset)

    test_size = len(train_clean_files) - train_size

    for clean_file, noisy_file in zip(
        train_clean_files[-test_size:],
        train_noisy_files[-test_size:]
    ):
        os.replace(clean_file, TEST_CLEAN_PATH / clean_file.name)
        os.replace(noisy_file, TEST_NOISY_PATH / noisy_file.name)



#Important to note, to run this script you need a python envioerment with python version 3.12 to avoid dependency conflicts
#This script was run once already to generate our data set, only run if you wish to generate a new dataset, since noise addition function is randomized
if __name__ == "__main__":
    generate_dataset()
    split_dataset(size_training_dataset=400)