import soundfile as sf
import random
import torchaudio
import torch

def load_audio_soundfile(file_path):
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

def normalize_tensors(clean_file:str, noise_files:list, target_sr:int, segment_length:int) -> tuple[torch.Tensor, torch.Tensor]:
    clean, sr = load_audio_soundfile(clean_file)

    if sr != target_sr:
        clean = torchaudio.functional.resample(clean, sr, target_sr)

    if clean.size(0) > 1:
        clean = clean.mean(dim=0, keepdim=True)

    if clean.size(1) > segment_length:
        start = random.randint(0, clean.size(1) - segment_length)
        clean = clean[:, start:start + segment_length]

    noise_file = random.choice(noise_files)
    noise, nsr = load_audio_soundfile(noise_file)

    if nsr != target_sr:
        noise = torchaudio.functional.resample(noise, nsr, target_sr)

    return noise, clean