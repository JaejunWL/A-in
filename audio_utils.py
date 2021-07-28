import io
import math
import warnings
from typing import Optional, Tuple, Callable

import torch
from torch import Tensor
from torchaudio._internal import module_utils as _mod_utils
import torchaudio
import torchaudio.transforms as T

import IPython.display as ipd

def _get_complex_dtype(real_dtype: torch.dtype):
    if real_dtype == torch.double:
        return torch.cdouble
    if real_dtype == torch.float:
        return torch.cfloat
    if real_dtype == torch.half:
        return torch.complex32
    raise ValueError(f'Unexpected dtype {real_dtype}')

def get_complex_spectrogram(waveform, n_fft = 2048, win_len = 2048, hop_len = 512, power=None, device='cuda'):
    spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_len,
    hop_length=hop_len,
    center=True,
    pad_mode="reflect",
    power=power,
    return_complex=0,
    )
    if device == 'cuda':
        spectrogram.cuda()
    return spectrogram(waveform)

def get_spectrogram(waveform, n_fft = 2048, win_len = 2048, hop_len = 512, power=2):
    spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_len,
    hop_length=hop_len,
    center=True,
    pad_mode="reflect",
    power=power,
    )
    return spectrogram(waveform)

def custom_griffinlim(specgram, input_angle, mask, window, n_fft, hop_length, win_length, power, n_iter,
                      momentum, length: Optional, angle_init='rand'):
    assert momentum < 1, 'momentum={} > 1 can be unstable'.format(momentum)
    assert momentum >= 0, 'momentum={} < 0'.format(momentum)

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    specgram = specgram.pow(1 / power)
    
    if input_angle != None and mask != None:
        _input_angle = input_angle
        _mask = mask
        
    # initialize the phase
    if angle_init == 'rand':
        angles = torch.rand(
            specgram.size(),
            dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
    elif angle_init == 'one':
        angles = torch.full(
            specgram.size(), 1,
            dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
    elif angle_init == 'zero':
        angles = torch.full(
            specgram.size(), 0,
            dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)


    # And initialize the previous iterate to 0
    tprev = torch.tensor(0., dtype=specgram.dtype, device=specgram.device)
    for _ in range(n_iter):
        angles = _input_angle*(1-_mask) + angles * _mask
        # Invert with our current estimate of the phases
        inverse = torch.istft(specgram * angles,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              win_length=win_length,
                              window=window,
                              length=length)

        # Rebuild the spectrogram
        rebuilt = torch.stft(
            input=inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Update our phase estimates
        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum / (1 + momentum))
        angles = angles.div(angles.abs().add(1e-16))

        # Store the previous iterate
        tprev = rebuilt

    # Return the final phase estimates
    waveform = torch.istft(specgram * angles,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           win_length=win_length,
                           window=window,
                           length=length
                          )

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform


class Custom_GriffinLim(torch.nn.Module):
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

    Implementation ported from
    :footcite:`brian_mcfee-proc-scipy-2015`, :footcite:`6701851` and :footcite:`1172092`.

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        n_iter (int, optional): Number of iteration for phase recovery process. (Default: ``32``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        momentum (float, optional): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge. (Default: ``0.99``)
        length (int, optional): Array length of the expected output. (Default: ``None``)
        rand_init (bool, optional): Initializes phase randomly if True and to zero otherwise. (Default: ``True``)
    """
    __constants__ = ['n_fft', 'n_iter', 'win_length', 'hop_length', 'power',
                     'length', 'momentum', 'rand_init']

    def __init__(self,
                 n_fft: int = 400,
                 n_iter: int = 32,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: float = 2.,
                 wkwargs: Optional[dict] = None,
                 momentum: float = 0.99,
                 length: Optional[int] = None,
                 rand_init: str = 'one') -> None:
        super(Custom_GriffinLim, self).__init__()

        assert momentum < 1, 'momentum={} > 1 can be unstable'.format(momentum)
        assert momentum >= 0, 'momentum={} < 0'.format(momentum)

        self.n_fft = n_fft
        self.n_iter = n_iter
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.length = length
        self.power = power
        self.momentum = momentum / (1 + momentum)
        self.rand_init = rand_init

    def forward(self, specgram: Tensor, input_angle: Tensor, mask: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor):
                A magnitude-only STFT spectrogram of dimension (..., freq, frames)
                where freq is ``n_fft // 2 + 1``.

        Returns:
            Tensor: waveform of (..., time), where time equals the ``length`` parameter if given.
        """
        return custom_griffinlim(specgram, input_angle, mask, self.window, self.n_fft, self.hop_length, self.win_length, self.power,
                            self.n_iter, self.momentum, self.length, self.rand_init)