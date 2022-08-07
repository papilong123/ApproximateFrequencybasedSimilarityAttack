import numpy as np
import pywt
import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# for evaluating
class DWT_audio:
    def __init__(self, wavename):
        self.wavename = wavename

    def __call__(self, signal, decomposition_level=1):
        signal = signal.detach().cpu()
        coeffs = []
        for i in range(len(signal)):
            single_coeffs = pywt.wavedec(signal[i], self.wavename, level=decomposition_level)
            coeffs.append(single_coeffs)
        # [ca3, cd3, cd2, cd1] = coeffs when decomposition_level is 3
        for i in range(len(signal)):
            for j in range(1, decomposition_level + 1):
                coeffs[i][j] = np.zeros_like(coeffs[i][j])
        return coeffs

# for evaluating
class IDWT_audio:
    def __init__(self, wavename):
        self.wavename = wavename

    def __call__(self, coeffs):
        rec_signal = []
        for i in range(len(coeffs)):
            single_audio = pywt.waverec(coeffs[i], self.wavename)
            rec_signal.append(single_audio)
        if isinstance(rec_signal, list):
            another = np.array(rec_signal)
            rec_signal = torch.from_numpy(another).to(device)
        return rec_signal


# for training
class DWT_audio_torch:
    def __init__(self, wave_name):
        self.wave_name = wave_name

    def __call__(self, signal, decomposition_level=1):
        signal = torch.unsqueeze(signal, 1).to('cpu')
        dwt = DWT1DForward(wave=self.wave_name, J=decomposition_level)
        cA, cD = dwt(signal)
        for i in range(len(cD)):
            cD[i][:, :] = 0
            cD[i].to(device)
        return cA.to(device), cD


# for training
class IDWT_audio_torch:
    def __init__(self, wave_name):
        self.wave_name = wave_name

    def __call__(self, cA, cD):
        idwt = DWT1DInverse(wave=self.wave_name)
        cA = cA.to('cpu')
        for i in range(len(cD)):
            cD[i].to('cpu')
        audio = idwt((cA, cD))
        audio = torch.squeeze(audio)
        return audio.to(device)
