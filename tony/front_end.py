""" Front-end: processing raw data input """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchaudio.functional as ta_F
import torchaudio


def magnitude(cplx_input, power=2., eps=1e-07):
    mag_summed = cplx_input.pow(power).sum(-1) + eps
    return mag_summed.pow(0.5)



class FrontEnd(nn.Module):
    def __init__(self, args, gpu_num=0):
        super(FrontEnd, self).__init__()
        self.eps = args.eps
        self.channel = args.channel
        self.n_fft = args.n_fft
        self.hop_length = self.n_fft//4 if args.hop_length==None else args.hop_length
        self.win_length = self.n_fft if args.win_length==None else args.win_length

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_num}")
        
        # spectrogram configs
        if args.window=="hann":
            self.window = torch.hann_window(window_length=self.win_length, periodic=True).to(device)
        elif args.window=="hamming":
            self.window = torch.hamming_window(window_length=self.win_length, periodic=True).to(device)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=args.sample_rate, \
                                                                    n_fft=self.n_fft, win_length=self.win_length, \
                                                                    hop_length=self.hop_length, n_mels=args.mel_channel_num).to(device)
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=args.sample_rate, \
                                                            n_mfcc=40, log_mels=True).to(device)

        # resampler
        self.resample_func = torchaudio.transforms.Resample(orig_freq=args.sample_rate, new_freq=16000) if args.sample_rate!=16000 else None


    def forward(self, input, mode):
        # front-end function which combines all demanded features channel-wise
        # input shape : batch x channel x raw waveform
        # output shape : batch x channel x frequency x time
        
        # handling mono channeled input
        if len(input.shape)==2:
            input = input.unsqueeze(1)

        front_output_list = []
        phase = None
        for cur_mode in mode:
            ''' STFT '''
            if cur_mode=="cplx" or "mag" in cur_mode:
                input_batchwise = input.reshape(input.shape[0]*input.shape[1], input.shape[2])
                cplx_batch = torch.stft(input_batchwise, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                # batch x channel x freq x time x cplx
                cplx = cplx_batch.reshape(input.shape[0], input.shape[1], cplx_batch.shape[1], cplx_batch.shape[2], cplx_batch.shape[3])

                # discard last time step
                if input.shape[2] % round(self.n_fft/4) == 0:
                    cplx = cplx[:, :, :, :-1]
                # discard highest frequency
                if self.n_fft % 2 == 0:
                    cplx = cplx[:, :, :-1]
                
                if cur_mode=="cplx":
                    # batch x channel x cplx x freq x time
                    front_output_list.append(cplx.permute(0, 1, 4, 2, 3))
                    continue
                
                ### magnitude
                # batch x freq x time
                mag = magnitude(cplx).squeeze(1)
                front_output_list.append(mag)
                ### phase
                if cur_mode=="mag_phase" and phase==None:
                    phase = torch.atan2(cplx[..., 1], cplx[..., 0])
            
                ''' Mel '''
            elif "mel" in cur_mode:
                # batch x mel bin x time
                mel_spec = self.mel_transform(input)
                mel_spec = mel_spec[..., :-1].squeeze(1)
                mel_spec = torch.log10(mel_spec+self.eps) if cur_mode=="log_mel" else mel_spec
                front_output_list.append(mel_spec)
            
                ''' MFCC '''
            elif "mfcc" in cur_mode:
                # batch x mfcc bin x time
                mfcc = self.self.mfcc_transform(input)
                mfcc = mfcc[..., :-1].squeeze(1)
                front_output_list.append(mfcc)


        # combine all demanded features
        if not front_output_list:
            raise NameError("NameError at FrontEnd: check using features for front-end")
        elif len(mode)!=1:
            front_output = front_output_list
        else:
            front_output = front_output_list[0]
            
        return front_output, phase





if __name__ == '__main__':
    from config import args
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ["CUDA_VISIBLE_DEVICES"] = args.using_gpus
    os.environ['MASTER_PORT'] = args.master_port

    batch_size = 16
    # segment_length = 512*128*6
    segment_length = args.segment_length
    input = torch.rand((batch_size, segment_length)).cuda()

    mode = ["cplx", "mag_phase", "mel"]
    fe = FrontEnd(args)
    
    output, out_phase = fe(input, mode=mode)
    # print(output.shape)
    for cur_output in output:
        print(cur_output.shape)

    print(out_phase.shape)




