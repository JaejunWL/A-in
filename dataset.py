import os
import cv2
import argparse
import librosa
import numpy as np

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Audio, display

import easydict
from sklearn.model_selection import train_test_split
ALLMASKTYPES = ['time', 'bbox', 'freeform']


class InpaintDataset(Dataset):
    def __init__(self, opt, split):
        self.opt = opt
        self.split = split
        self.pow_to_db = torchaudio.transforms.AmplitudeToDB('power')
        self.get_list()
        
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
        # spec = self.get_comlex_spectrogram(audio)
        spec = self.get_spectrogram(audio).unsqueeze(-1)
        spec = self.pow_to_db(spec)
        spec = spec.squeeze(0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()

        phase = 0
        if self.opt.phase == 1:
            complex_spec = self.get_complex_spectrogram(audio)
            complex_spec_comp = torch.view_as_complex(complex_spec)
            phase = torch.angle(complex_spec_comp)
            phase_np = np.asarray(phase)
            phase_unwrapped = np.unwrap(phase_np)
            phase = torch.tensor(phase_unwrapped).float()
            spec = torch.cat([spec, phase], 0)

        if self.opt.mask_init == 'lerp':
            lerp_mask = self.make_lerp_mask(spec, mask)
            return audio, spec, mask, lerp_mask
        else:
            return audio, spec, mask, mask

    
    def get_list(self):
        margs_trainvalid = np.loadtxt('../split/margs_trainvalid.txt', delimiter=',', dtype=str)
        margs_test = np.loadtxt('../split/margs_test.txt', delimiter=',', dtype=str)
        nuss_trainvalid = np.loadtxt('../split/nuss_trainvalid.txt', delimiter=',', dtype=str)
        nuss_test = np.loadtxt('../split/nuss_test.txt', delimiter=',', dtype=str)
        vocals_trainvalid = np.loadtxt('../split/vocals_trainvalid.txt', delimiter=',', dtype=str)
        vocals_test = np.loadtxt('../split/vocals_test.txt', delimiter=',', dtype=str)

        margs_train, margs_valid = train_test_split(margs_trainvalid, test_size=0.1, shuffle=False, random_state=21)
        nuss_train, nuss_valid = train_test_split(nuss_trainvalid, test_size=0.1, shuffle=False, random_state=21)
        vocals_train, vocals_valid = train_test_split(vocals_trainvalid, test_size=0.1, shuffle=False, random_state=21)

        if self.split == 'TRAIN':
            train_list = self.dataset_merge(list(margs_train) + list(nuss_train) + list(vocals_train))
            self.fl = train_list
        elif self.split == 'VALID':
            valid_list = self.dataset_merge(list(margs_valid) + list(nuss_valid) + list(vocals_valid))
            self.fl = valid_list
        elif self.split == 'TEST':
            test_list = self.dataset_merge(list(margs_test) + list(nuss_test) + list(vocals_test))
            self.fl = test_list
    
    def dataset_merge(self, dataset):
        merged_dataset = []
        for ix, lis in enumerate(dataset):
            dat = torchaudio.info(os.path.join(self.opt.data_dir, lis))
            if dat.num_frames > 44100*2.5:
                merged_dataset.append(lis)
        return merged_dataset

    def get_audio(self, index):
        fn = self.fl[index]
        audio_path = os.path.join(self.opt.data_dir, fn)
        num_frames = torchaudio.info(audio_path).num_frames
        if num_frames <= self.opt.input_length:
            audio, sr = torchaudio.load(audio_path)
            audio = torch.nn.functional.pad(audio, (0, self.opt.input_length-num_frames), mode='constant', value=0)
        else:
            random_idx = np.random.randint(num_frames - self.opt.input_length)
            audio, sr = torchaudio.load(audio_path, frame_offset=random_idx, num_frames=self.opt.input_length)
        return audio

    def get_valid_audio(self, index):
        fn = self.fl[index]
        audio_path = os.path.join(self.opt.data_dir, fn)
        num_frames = torchaudio.info(audio_path).num_frames
        if num_frames <= self.opt.input_length:
            audio, sr = torchaudio.load(audio_path)
            audio = torch.nn.functional.pad(audio, (0, self.opt.input_length-num_frames), mode='constant', value=0)
        else:
            audio, sr = torchaudio.load(audio_path, frame_offset=0, num_frames=self.opt.input_length)
        return audio

    def get_complex_spectrogram(self, waveform, n_fft = 2048, win_len = 2048, hop_len = 512, power=None):
        spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
        return_complex=0,
        )
        return spectrogram(waveform)

    def get_spectrogram(self, waveform, n_fft = 2048, win_len = 2048, hop_len = 512, power=2):
        spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
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

    def make_lerp_mask(self, spec, mask):
        spec_pad = torch.nn.functional.pad(spec, (1, 1, 0, 0), mode='constant', value=0)
        mask_range = torch.where(mask[:,0,:]==1)
        start = torch.min(mask_range[-1])
        end = torch.max(mask_range[-1])
        merged = torch.cat([spec_pad[:,:,start], spec_pad[:,:,end+2]], 0)
        lerp = torch.nn.functional.interpolate(merged.permute(1, 0).unsqueeze(0), size=end-start+1, mode='linear')
        lerp_mask = torch.zeros(mask.shape)
        lerp_mask[:, :, start:end+1] = lerp
        return lerp_mask
        
    def __len__(self):
        return len(self.fl)