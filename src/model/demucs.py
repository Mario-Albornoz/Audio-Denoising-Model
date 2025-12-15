import numpy as np
import torch
import torch.nn as nn
from src.utils.resample import *

class BottleNeckLTSM(nn.Module):
    def __init__(self, dim, bi_directional:bool=False, layers:int = 2):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, layers, False, False, 0.0, bi_directional)

    def forward(self, sample, hidden=None):
        sample, hidden = self.lstm(sample, hidden)
        return sample, hidden


class BaseDemucs(nn.Module):
    def __init__(
            self,
            layers:int = 5,
            channels_in:int = 1 ,
            channels_out:int = 1,
            kernel_size:int = 8,
            hidden_channels:int = 48 ,
            max_hidden_channels:int = 10000,
            stride : int = 4,
            depth : int = 5,
            normalize: bool = True,
            up_scale_factor : int = 2,
            resmaple :int = 1,
            floor = 1e-3,
    ):

        super().__init__()
        self.layer = layers
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size =  kernel_size
        self.hidden_channels = hidden_channels
        self.max_hidden_channels = max_hidden_channels
        self.stride = stride
        self.depth = depth
        self.normalize = normalize
        self.up_scale_factor = up_scale_factor
        self.floor = floor
        self.resample = resmaple

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        channel_scale = 2 #double feature map per convolutional layer because PReLu halfs the size of our channels

        for index in range(layers):
            decoder_target_out = channels_in
            encode = []
            encode += [
                nn.Conv1d(in_channels=channels_in, out_channels=hidden_channels, kernel_size=kernel_size,
                          stride=stride),
                nn.PReLU(),
                nn.Conv1d(hidden_channels, hidden_channels * channel_scale, 1),
                nn.GLU(dim=1)
            ]
            self.encoder.append(nn.Sequential(*encode))
            encoder_out_channels = hidden_channels * channel_scale // 2
            decode = []
            decode += [
                nn.Conv1d(in_channels=encoder_out_channels, out_channels=hidden_channels * channel_scale, kernel_size=1,
                          stride=1),
                nn.GLU(dim=1),
                nn.ConvTranspose1d(hidden_channels, decoder_target_out, kernel_size, stride)
            ]
            if index < layers - 1:
                decode.append(nn.ReLU())

            self.decoder.append(nn.Sequential(*decode))

            channels_in = encoder_out_channels
            hidden_channels = min(int(2 * hidden_channels), max_hidden_channels)

        self.lstm  = BottleNeckLTSM(encoder_out_channels)


    def valid_length(self, length):
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def return_hidden_lstm_state(self):
        return self.lstm

    def forward(self, mix:torch.Tensor):
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (std + self.floor)
        else:
            std = 1

        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        length = mix.shape[-1]
        sample = mix
        sample = F.pad(sample, (0, self.valid_length(length) - length))

        if self.resample > 1:
            for idx in range(0, self.resample//2):
                sample = F.interpolate(sample, scale_factor=2, mode='linear', align_corners=False)

        skips = []
        for encoder in self.encoder:
            sample = encoder(sample)
            skips.append(sample)

        sample = sample.permute(2,0,1)
        sample, _ = self.lstm(sample)
        sample = sample.permute(1,2,0)

        for decoder in reversed(self.decoder):
            skip = skips.pop(-1)
            sample = sample + skip[..., :sample.shape[-1]]
            sample = decoder(sample)

        if self.resample > 1:
            for idx in range(0, self.resample//2):
                sample = F.interpolate(sample, scale_factor=0.5, mode='linear', align_corners=False)

        sample = sample[..., :length]

        return sample * std

class RealTimeDemucs(nn.Module):
    def __init__(self, model:BaseDemucs = BaseDemucs):
        super().__init__()
        self.model = model
        self.lstm_hidden_state = None

    def forward(self, mix: torch.Tensor):
        sample = mix
        if self.model.normalize:
            mono = mix.mean(dim= 1, keepdim=True)
            std = mono.std(dim=1, keepdim=True)
            sample = sample / (std + self.model.floor)
        else:
            std = 1

        length = mix.shape[-1]
        sample = mix
        sample = F.pad(sample, (0, self.model.valid_length(length) - length))

        skips = []
        for encode in self.model.encoder:
            sample = encode(sample)
            skips.append(sample)

        sample = sample.permute(2, 0, 1)
        sample, self.lstm_hidden_state = self.model.lstm(sample, self.lstm_hidden_state)
        sample = sample.permute(1, 2, 0)

        for decoder in reversed(self.model.decoder):
            skip = skips.pop(-1)
            sample = sample + skip[:, :sample.shape[-1]]
            sample = decoder(sample)

        return sample * std

    def reset(self):
        self.lstm_hidden_state = None

class ProcessRealTime:
    def __init__(self, trained_model_path: str,  base_model: BaseDemucs = BaseDemucs, device: str = 'cpu'):
        self.base_model = base_model
        self.device = device
        self.base_model.load_state_dict(torch.load(trained_model_path, map_location=device))
        self.base_model.eval()

        self.model = RealTimeDemucs(model= base_model).to(device)

    @torch.no_grad()
    def process_batch(self, audio:np.ndarray) -> np.ndarray:
        data: torch.Tensor = torch.from_numpy(audio)

        if data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.dim() == 2:
            data = data.unsqueeze(0)

        enhanced: torch.Tensor = self.model(data)

        return enhanced.squeeze(0).cpu().numpy()

    def reset(self):
        self.model.reset()
