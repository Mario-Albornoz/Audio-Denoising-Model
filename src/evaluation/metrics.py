import torch
import torch.nn as nn


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for music denoising with perceptual weighting
    """

    def __init__(
            self,
            n_ffts=None,
            spectral_weight: float = 0.5,  # Increased for music
            use_mel_scale: bool = True,
            use_log_magnitude: bool = True,
            perceptual_weighting: bool = True,
            sample_rate: int = 16000
    ):
        super().__init__()

        # Larger FFT sizes for music (captures lower frequencies better)
        if n_ffts is None:
            n_ffts = [2048, 4096, 8192]  # Changed from [512, 1024, 2048]

        self.n_ffts = n_ffts
        self.spectral_weight = spectral_weight
        self.use_mel_scale = use_mel_scale
        self.use_log_magnitude = use_log_magnitude
        self.perceptual_weighting = perceptual_weighting
        self.sample_rate = sample_rate

        self.L1 = nn.L1Loss()

        # Pre-register Hann windows to avoid recreating them
        for n_fft in n_ffts:
            self.register_buffer(f'window_{n_fft}', torch.hann_window(n_fft))

        # Mel filterbanks for perceptual weighting
        if use_mel_scale:
            self.mel_filters = {}
            for n_fft in n_ffts:
                n_mels = min(128, n_fft // 2)  # Adaptive mel bins
                self.mel_filters[n_fft] = self._create_mel_filterbank(
                    n_fft, n_mels, sample_rate
                )

    def _create_mel_filterbank(self, n_fft, n_mels, sample_rate):
        """Create mel filterbank for perceptual weighting"""
        # Simplified mel filterbank creation
        mel_basis = torch.zeros(n_mels, n_fft // 2 + 1)

        # Linear spacing in mel scale
        mel_min = self._hz_to_mel(0)
        mel_max = self._hz_to_mel(sample_rate / 2)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

        # Create triangular filters
        for i in range(n_mels):
            left, center, right = bin_points[i:i + 3]
            # Rising slope
            mel_basis[i, left:center] = (torch.arange(left, center).float() - left) / (center - left)
            # Falling slope
            mel_basis[i, center:right] = (right - torch.arange(center, right).float()) / (right - center)

        return mel_basis

    @staticmethod
    def _hz_to_mel(hz):
        # Convert to tensor if it's a scalar
        if not isinstance(hz, torch.Tensor):
            hz = torch.tensor(hz, dtype=torch.float32)
        return 2595 * torch.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel):
        # Convert to tensor if it's a scalar
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel, dtype=torch.float32)
        return 700 * (10 ** (mel / 2595) - 1)

    def _compute_spectral_loss(self, pred, target, n_fft):
        """Compute spectral loss for a given FFT size"""
        hop_length = n_fft // 4
        window = getattr(self, f'window_{n_fft}').to(pred.device)

        # Compute STFT
        pred_spec = torch.stft(
            pred.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,  # Add padding for better edge handling
            normalized=False
        )

        target_spec = torch.stft(
            target.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
            normalized=False
        )

        # Get magnitude
        pred_mag = pred_spec.abs()
        target_mag = target_spec.abs()

        # Apply mel-scale weighting if enabled
        if self.use_mel_scale and n_fft in self.mel_filters:
            mel_filter = self.mel_filters[n_fft].to(pred.device)
            # Apply mel filterbank: (batch, freq, time) @ (n_mels, freq).T
            pred_mag = torch.matmul(pred_mag.transpose(1, 2), mel_filter.T).transpose(1, 2)
            target_mag = torch.matmul(target_mag.transpose(1, 2), mel_filter.T).transpose(1, 2)

        # Log magnitude for perceptual similarity (music is perceived logarithmically)
        if self.use_log_magnitude:
            pred_mag = torch.log1p(pred_mag)  # log(1 + x) for numerical stability
            target_mag = torch.log1p(target_mag)

        # Compute L1 loss in spectral domain
        mag_loss = self.L1(pred_mag, target_mag)

        # Phase loss (important for music quality)
        if self.perceptual_weighting:
            pred_phase = pred_spec.angle()
            target_phase = target_spec.angle()
            # Phase difference loss (wrapped)
            phase_diff = torch.abs(torch.angle(torch.exp(1j * (pred_phase - target_phase))))
            phase_loss = phase_diff.mean()

            # Combine magnitude and phase (70% magnitude, 30% phase)
            return 0.7 * mag_loss + 0.3 * phase_loss

        return mag_loss

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted clean audio (batch, channels, samples)
            target: Target clean audio (batch, channels, samples)

        Returns:
            total_loss, time_loss, spectral_loss
        """
        # Ensure 3D input
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        # Time-domain loss
        time_loss = self.L1(pred, target)

        # Multi-scale spectral loss
        spectral_loss = 0.0
        num_scales = len(self.n_ffts)

        for n_fft in self.n_ffts:
            spectral_loss += self._compute_spectral_loss(pred, target, n_fft)

        spectral_loss = spectral_loss / num_scales

        # Combined loss (spectral weighted higher for music)
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
