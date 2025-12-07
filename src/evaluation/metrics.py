import torch
import torch.nn as nn

class MultiScaleLoss(nn.Module):

    def __init__(self, n_ffts=None, spectral_weight: float = 0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if n_ffts is None:
            n_ffts = [512, 1024, 2048]
        self.n_ffts = n_ffts
        self.spectral_weight = spectral_weight
        self.L1 = nn.L1Loss()

    def forward(self, pred, target):
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        time_loss = self.L1(pred, target)

        num_scales = len(self.n_ffts)
        spectral_loss = 0.0

        for n_fft in self.n_ffts:
            hop_length = n_fft // 4
            self.register_buffer(f'window_{n_fft}', torch.hann_window(n_fft))

            pred_spectogram = torch.stft(
                pred.squeeze(1),
                n_fft= n_fft,
                hop_length = hop_length,
                win_length=n_fft,
                window=torch.hann_window(n_fft).to(pred.device),
                return_complex=True
            ).abs()

            target_spectogram = torch.stft(
                target.squeeze(1),
                n_fft = n_fft,
                hop_length = hop_length,
                win_length=n_fft,
                window=torch.hann_window(n_fft).to(pred.device),
                return_complex=True
            ).abs()

            spectral_loss += self.L1(pred_spectogram, target_spectogram)

        spectral_loss = spectral_loss / num_scales
        total_loss = (1 - self.spectral_weight) * time_loss + \
                     self.spectral_weight * spectral_loss

        return total_loss, time_loss, spectral_loss


def compute_snr(clean : torch.Tensor, enhanced : torch.Tensor, eps = 1e-8):
    #flatten time dimension
    noise = enhanced - clean

    #-1 and -2 represent the last dimensions of the file (time (sample_rate) and chanels)
    clean_power = torch.mean(clean ** 2, dim=(-1,-2)) if clean.dim() > 1 else torch.mean(clean**2)
    noise_power = torch.mean(noise ** 2, dim=(-1, -2)) if noise.dim() > 1 else torch.mean(noise ** 2)

    snr = 10.0 * torch.log10((clean_power + eps)/(noise_power + eps))

    return snr

def compute_lsd(clean : torch.Tensor, enhanced : torch.Tensor, n_fft = 512,eps = 1e-8) -> float:
    if clean.dim() == 2:
        clean = clean.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)
    elif clean.dim() == 3:
        clean = clean.squeeze(1)
        enhanced = enhanced.squeeze(1)

    S_clean = torch.stft(clean, n_fft=n_fft, hop_length=n_fft // 4, return_complex=True)
    S_enh = torch.stft(enhanced, n_fft=n_fft, hop_length=n_fft // 4, return_complex=True)
    mag_clean = torch.abs(S_clean) + eps
    mag_enh = torch.abs(S_enh) + eps

    log_clean = 10 * torch.log10(mag_clean ** 2 + eps)
    log_enh = 10 * torch.log10(mag_enh ** 2 + eps)

    lsd_frame = ((log_clean - log_enh) ** 2).mean(dim=(-1, -2)).sqrt()
    return float(lsd_frame.mean().item())


def si_sdr_loss(estimation, reference, eps=1e-8) -> torch.Tensor  :
    if estimation.dim() == 2:
        estimation = estimation.unsqueeze(0)
        reference = reference.unsqueeze(0)

    s = reference.view(reference.size(0), -1)
    s_hat = estimation.view(estimation.size(0), -1)

    s_zm = s - s.mean(dim=1, keepdim=True)
    shat_zm = s_hat - s_hat.mean(dim=1, keepdim=True)

    proj = (torch.sum(shat_zm * s_zm, dim=1, keepdim=True) * s_zm) / (torch.sum(s_zm ** 2, dim=1, keepdim=True) + eps)
    e_noise = shat_zm - proj

    ratio = torch.sum(proj ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps)
    si_sdr = 10 * torch.log10(ratio + eps)

    return -torch.mean(si_sdr)
