import random

import soundfile as sf
import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path


class DenoisingDataSet(Dataset):
    def __init__(
            self,
            noisy_dir: str,
            clean_dir: str,
            target_sr: int = 16000,
            segment_length: int = 80000,
            use_augmentation: bool = False
    ):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.stems: list[str] = []
        self.clear_unmateched_file()
        self.use_augmentation = use_augmentation


    def __len__(self):
        return len(self.noisy_files)

    def clear_unmateched_file(self):
        noise_files = list(self.noisy_dir.glob("*.wav"))
        raw_files = list(self.clean_dir.glob("*.wav"))

        print(len(noise_files), len(raw_files))

        noise_stems = {f.stem for f in noise_files}
        raw_stems = {f.stem for f in raw_files}

        common_stems = noise_stems & raw_stems

        self.noisy_files = sorted([f for f in noise_files if f.stem in common_stems])
        self.clean_files = sorted([f for f in raw_files if f.stem in common_stems])
        print(len(self.noisy_files), len(self.clean_files))

        if len(noise_files) != len(self.noisy_files):
            print(f"Filtered out {len(noise_files) - len(self.noisy_files)} noisy files without matching clean files")
        if len(raw_files) != len(self.clean_files):
            print(f"Filtered out {len(raw_files) - len(self.clean_files)} clean files without matching noisy files")

        if len(self.noisy_files) != len(self.clean_files):
            print("Noisy file count ", len(self.noisy_files), "clean file count: ", len(self.clean_files))
            raise ValueError("Mismatch in dataset length: noisy vs clean file counts")

        self.stems = [p.stem for p in self.noisy_files]

    def _load_and_resample(self, path: Path ):
        data, sr = sf.read(str(path), dtype='float32')
        waveform = torch.from_numpy(data)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (samples,) -> (1, samples)
        else:
            waveform = waveform.t()  # (samples, channels) -> (channels, samples)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    def _augment(self, noisy, clean):
        if random.random() > 0.5:
            gain = random.uniform(0.7,0.13)
            noisy = gain * noisy
            clean = gain * clean

        if random.random() > 0.5:
            shift = random.randint(-1000,1000)
            noisy = torch.roll(noisy, shifts=shift, dims=-1)
            clean = torch.roll(clean, shifts=shift, dims=-1)

        if random.random() > 0.5:
            extra_noise = torch.randn_like(noisy)
            noisy += extra_noise

        return noisy, clean

    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_files[idx]

        noisy = self._load_and_resample(noisy_path)
        clean = self._load_and_resample(clean_path)

        min_len = min(noisy.size(1), clean.size(1))
        noisy = noisy[:, :min_len]
        clean = clean[:, :min_len]

        if min_len > self.segment_length:
            start = random.randint(0, min_len - self.segment_length)
            noisy = noisy[:, start:start + self.segment_length]
            clean = clean[:, start:start + self.segment_length]

        if noisy.size(1) < self.segment_length:
            pad = self.segment_length - noisy.size(1)
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))

        if self.use_augmentation:
            self._augment(noisy, clean)

        return {
            "noisy": noisy,
            "clean": clean,
            "stem": noisy_path.stem
        }