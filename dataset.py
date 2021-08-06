import os
import cv2
import argparse
import librosa
import numpy as np
import easydict
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset

import audio_utils

ALLMASKTYPES = ['time', 'bbox', 'freeform']

class InpaintDataset(Dataset):
    def __init__(self, opt, split):
        self.opt = opt
        self.split = split
        self.get_list()
        if opt.spec_pow == 1:
            self.linear_to_db = torchaudio.transforms.AmplitudeToDB('magnitude')
        elif opt.spec_pow == 2:
            self.linear_to_db = torchaudio.transforms.AmplitudeToDB('power')
        
    def __getitem__(self, index):
        if self.split == 'TRAIN':
            if self.opt.mask_type == 'time':
                mask = self.time_mask()
            if self.opt.mask_type == 'bbox':
                mask = self.bbox2mask()
            if self.opt.mask_type == 'freeform':
                mask = self.random_ff_mask()
            audio = self.get_audio(index)
        elif self.split in ['VALID', 'TEST']:
            if self.opt.mask_type == 'time':
                masks_dir = '../split/fixedmask_time_2048'
                mask = np.load(os.path.join(masks_dir, str(index)) + '.npy')
            if self.opt.mask_type == 'bbox':
                masks_dir = '../split/fixedmask_bbox_2048'
                mask = np.load(os.path.join(masks_dir, str(index)) + '.npy')
            if self.opt.mask_type == 'freeform':
                masks_dir = '../split/fixedmask_freeform_2048'
                mask = np.load(os.path.join(masks_dir, str(index)) + '.npy')
            audio = self.get_valid_audio(index)
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        mask_init = mask.clone()

        complex_spec = self.get_spectrogram(audio, power=None, return_complex=1)
        mag_spec = torch.abs(complex_spec)
        mag_spec = mag_spec ** self.opt.spec_pow

        if self.opt.mask_init == 'lerp':
            mask_init = self.make_lerp_mask(mag_spec[0:1], mask_init)
            mask_init = self.linear_to_db(mask_init) * mask
            spec = self.linear_to_db(mag_spec)
            if self.opt.phase == 1:
                phase_spec = torch.angle(complex_spec)
                mask_init = torch.cat([mask_init, mask], 0)
                spec = torch.cat([spec, phase_spec], 0)
        else:
            spec = self.linear_to_db(mag_spec)
            if self.opt.phase == 1:
                phase_spec = torch.angle(complex_spec)
                spec = torch.cat([spec, phase_spec], 0)

        if self.opt.pos_enc != None:
            pos_enc = self.get_poistional_encoding(self.opt.image_height, 44100/2, scale=self.opt.pos_enc)
            spec = torch.cat([spec, pos_enc], 0)

        return audio, spec, mask, mask_init

    def get_list(self):
        merged_dataset = []
        if self.split == 'TRAIN':
            margs_train = np.loadtxt('../split/margs_trainvalid.txt', delimiter=',', dtype=str)
            nuss_train = np.loadtxt('../split/nuss_trainvalid.txt', delimiter=',', dtype=str)
            vocals_train = np.loadtxt('../split/vocals_trainvalid.txt', delimiter=',', dtype=str)
            merged_dataset += list(margs_train) + list(nuss_train) + list(vocals_train)
            if len(self.opt.add_datasets) > 0:
                for add_dataset in self.opt.add_datasets:
                    add_list = glob.glob(os.path.join(add_dataset, '*', '*.wav'))
                    merged_dataset += add_list
        elif self.split == 'VALID':
            margs_test = np.loadtxt('../split/margs_test.txt', delimiter=',', dtype=str)
            nuss_test = np.loadtxt('../split/nuss_test.txt', delimiter=',', dtype=str)
            vocals_test = np.loadtxt('../split/vocals_test.txt', delimiter=',', dtype=str)
            merged_dataset += list(margs_test) + list(nuss_test) + list(vocals_test)
        self.fl = merged_dataset

    def get_audio(self, index):
        fn = self.fl[index]
        audio_path = os.path.join(self.opt.data_dir, fn)
        num_frames = torchaudio.info(audio_path).num_frames
        if num_frames > self.opt.input_length:
            random_idx = np.random.randint(num_frames - self.opt.input_length)
            audio, sr = torchaudio.load(audio_path, frame_offset=random_idx, num_frames=self.opt.input_length)
        else:
            div = np.min([self.opt.input_length // num_frames, 3])
            audio, sr = torchaudio.load(audio_path)
            temp = audio.clone()
            for i in range(div-1):
                audio = torch.cat([audio, temp], -1)
            audio = torch.nn.functional.pad(audio, (0, self.opt.input_length-num_frames*div), mode='constant', value=0)
        return audio

    def get_valid_audio(self, index):
        fn = self.fl[index]
        audio_path = os.path.join(self.opt.data_dir, fn)
        num_frames = torchaudio.info(audio_path).num_frames
        if num_frames > self.opt.input_length:
            audio, sr = torchaudio.load(audio_path, frame_offset=0, num_frames=self.opt.input_length)
        else:
            div = np.min([self.opt.input_length // num_frames, 3])
            audio, sr = torchaudio.load(audio_path)
            temp = audio.clone()
            for i in range(div-1):
                audio = torch.cat([audio, temp], -1)
            audio = torch.nn.functional.pad(audio, (0, self.opt.input_length-num_frames*div), mode='constant', value=0)
        return audio

    def get_spectrogram(self, waveform, n_fft = 2048, win_len = 2048, hop_len = 512, power=2, return_complex=0):
        spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
        return_complex=return_complex,
        )
        return spectrogram(waveform)

    def random_bbox(self):
        max_freq_ix = self.opt.image_height - self.opt.bbox_shape
        max_frame_ix = self.opt.image_width - self.opt.bbox_shape

        box_freq_ix = np.random.randint(max_freq_ix)
        box_frame_ix = np.random.randint(max_frame_ix)

        return (box_freq_ix, box_frame_ix, self.opt.bbox_shape, self.opt.bbox_shape)

    def time_mask(self):
        mask_width = np.random.randint(low=8, high=87)
        max_frame_ix = self.opt.image_width - mask_width
        t = np.random.randint(max_frame_ix)
        mask = np.zeros((self.opt.image_height, self.opt.image_width))
        mask[:,t:t+mask_width] = 1
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    def bbox2mask(self):
        bboxs = []
        times = np.random.randint(8)
        for i in range(times):
            bbox = self.random_bbox()
            bboxs.append(bbox)
        mask = np.zeros((self.opt.image_height, self.opt.image_width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    def random_ff_mask(self):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        mask = np.zeros((self.opt.image_height, self.opt.image_width), np.float32)
        max_angle = 4
        max_len = 200
        max_width = 20
        times = np.random.randint(7)
        for i in range(times):
            start_x = np.random.randint(self.opt.image_width)
            start_y = np.random.randint(self.opt.image_height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    # initial training point with linear interpolated mask (for opt.mask_init='lerp')
    def make_lerp_mask(self, spec, mask):
        spec_pad = torch.nn.functional.pad(spec, (1, 1, 0, 0), mode='constant', value=0)
        mask_range = torch.where(mask[:,0,:]==1)
        start = torch.min(mask_range[-1])
        end = torch.max(mask_range[-1])
        merged = torch.cat([spec_pad[:,:,start], spec_pad[:,:,end+2]], 0)
        lerp = torch.nn.functional.interpolate(merged.permute(1, 0).unsqueeze(0), size=end-start+1, mode='linear', align_corners=True)
        lerp_mask = torch.zeros(mask.shape)
        lerp_mask[:, :, start:end+1] = lerp
        return lerp_mask
        
    def get_poistional_encoding(self, freq_bin, max_freq, scale='cartesian'):
        freq = np.linspace(0, max_freq, freq_bin)
        if scale == 'mel':
            freq = 2595 * np.log10(1+freq/700)
        freq = freq / np.max(freq)
        freq = torch.from_numpy(freq.astype(np.float32)).contiguous()
        freq = freq.reshape(1, -1, 1)
        freq = freq.expand(1, self.opt.image_height, self.opt.image_width)
        return freq

    def __len__(self):
        return len(self.fl)