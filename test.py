import os
import numpy as np
import librosa 

import dataset
import easydict
import argparse
import network
import utils
import audio_utils

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import IPython.display as ipd

import warnings
from tqdm import tqdm

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--load_folder', type = str, default = 'test', help = 'saving path that is a folder')
    parser.add_argument('--load_model_folder', type = str, default = 'test', help = 'saving path that is a sub-folder')
    parser.add_argument('--data_dir', type = str, default = '/data1/singing_inpainting/dataset/testset', help = 'dataset directory')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--gpu_ids', type = str, default = "6", help = '')
    parser.add_argument('--epoched', type = int, default = None, help = 'which epoch model you want to use')
    parser.add_argument('--batch_sized', type = int, default = 4, help = 'A batch size of the model you want to load')
    # Test audio parameters
    parser.add_argument('--test_audio', type=str, default = '', help = 'path of the test audio')
    parser.add_argument('--audio_start', type=float, default=0, help = 'start time of the audio')
    parser.add_argument('--audio_end', type=int, default=None, help = 'start time of the audio')
    parser.add_argument('--mask_xs', type=float, default=None, help = 'mask start time')
    parser.add_argument('--mask_xe', type=float, default=None, help = 'mask end time')
    parser.add_argument('--mask_ys', type=int, default=None, help = 'mask start freq')
    parser.add_argument('--mask_ye', type=int, default=None, help = 'mask end freq')
    # Dataset parameters
    parser.add_argument('--mask_type', type = str, default = 'time', help = 'mask type')
    parser.add_argument('--mask_init', type = str, default = 'lerp', help = 'mask initialie point')
    parser.add_argument('--image_height', type = int, default = 1025, help = 'height of image')
    parser.add_argument('--image_width', type = int, default = 431, help = 'width of image')
    # parser.add_argument('--image_width', type = int, default = 259, help = 'width of image')
    parser.add_argument('--input_length', type = int, default = 220500, help = 'input length (sample)')
    # parser.add_argument('--input_length', type = int, default = 132300, help = 'input length (sample)')
    parser.add_argument('--spec_pow', type=int, default=2, help='1 for amplitude spec, 2 for power spec')
    parser.add_argument('--phase', type=int, default = 0, help = 'whether give phase information or not')
    parser.add_argument('--bbox', type = bool, default = False, help = 'whether use bbox shaping or not')
    # Network parameters
    parser.add_argument('--stage_num', type = int, default = 1, help = 'two stage method or just only stage')
    parser.add_argument('--in_channels', type = int, default = 2, help = 'input mag + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 1, help = 'output mag')
    parser.add_argument('--latent_channels', type = int, default = 32, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--pos_enc', type=str, default=None, help = 'positinoal embedding')
    parser.add_argument('--perceptual', type=str, default=None, help='whether use perceptual loss or not')
    # Other parameters
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--scale', type = float, default = 1, help = 'number of cpu threads to use during batch generation')

    opt = parser.parse_args()

    opt.load_dir = os.path.join('/data2/personal/jaejun/inpainting/results', opt.load_folder, opt.load_model_folder, 'models')
    opt.model_name = 'deepfillv2_WGAN_epoch' + str(opt.epoched) + '_batchsize' + str(opt.batch_sized) + '.pth'
    opt.result_dir = os.path.join('/data2/personal/jaejun/inpainting/results', opt.load_folder, opt.load_model_folder, 'test_' + str(opt.epoched), os.path.basename(opt.test_audio)[:-4])
    if opt.phase == 1:
        opt.in_channels = opt.in_channels + 1
        opt.out_channels = opt.out_channels + 1
    if opt.pos_enc != None:
        opt.in_channels = opt.in_channels + 1
    print(opt)

    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    save_dir = os.path.join(opt.result_dir)
    if opt.mask_ys != None and opt.mask_ye != None:
        opt.sample_name = str(np.round(0.5 * (opt.mask_xs + opt.mask_xe), 2)) + '_box'
    else:
        opt.sample_name = str(np.round(0.5 * (opt.mask_xs + opt.mask_xe), 2))

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------
    testset = dataset.InpaintDataset(opt, split='TEST')
    print('The overall number of images equals to %d' % len(testset))
    dataloader = DataLoader(testset, batch_size = 1, pin_memory = True, num_workers=opt.num_workers)
    
    # Define the dataloader
    # dataloader = DataLoader(testset, batch_size = 1, pin_memory = True, num_workers=opt.num_workers)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    model = network.GatedGenerator(opt)
    model_dir = os.path.join(opt.load_dir, opt.model_name)
    model.load_state_dict(torch.load(model_dir))
    model.cuda()

    L1Loss = nn.L1Loss()
    bw_L1Loss = nn.L1Loss(reduction='none')

    torch_gflim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=100, win_length=2048, hop_length=512, power=opt.spec_pow)
    torch_gflim.cuda()
    custom_gflim = audio_utils.Custom_GriffinLim(n_fft=2048, n_iter=100, win_length=2048, hop_length=512, power=opt.spec_pow, rand_init='pred')
    custom_gflim.cuda()

    # if opt.phase == 1:
        # custom_gflim_phase = audio_utils.Custom_GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512, power=opt.spec_pow, rand_init='pred')
        # custom_gflim_phase.cuda()

    # def db_to_linear(db, ref=1.0, power=1):
        # return 10.0**(db/10.0) * torch.tensor(ref)
    
    def db_to_linear(db_input, opt):
        if opt.spec_pow == 2:
            linear_output = torchaudio.functional.DB_to_amplitude(db_input, ref=1.0, power=1)
        elif opt.spec_pow == 1:
            linear_output = torchaudio.functional.DB_to_amplitude(db_input, ref=1.0, power=0.5)
        return linear_output

    with torch.no_grad():
        audio, img, mask, mask_init, mask_lerp, mask_start, mask_end = testset[0]
        print(audio.min(), audio.max())

        original_audio = audio.clone()

        audio = audio / opt.scale
        audio = audio.unsqueeze(0)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        mask_init = mask_init.unsqueeze(0)
        mask_lerp = mask_lerp.unsqueeze(0)

        img = img[:,:,:opt.image_height-1,:opt.image_width-3]
        mask = mask[:,:,:opt.image_height-1,:opt.image_width-3]
        mask_init = mask_init[:,:,:opt.image_height-1,:opt.image_width-3]
        mask_lerp = mask_lerp[:,:,:opt.image_height-1,:opt.image_width-3]

        img = img.cuda()
        mask = mask.cuda()
        mask_init = mask_init.cuda()
        mask_lerp = mask_lerp.cuda()

        second_out = model(img, mask, mask_init)

        gt = db_to_linear(img[0,0,:,:], opt)
        mask = mask[0,0,:,:]
        if opt.mask_init == 'lerp':
            mask_init = db_to_linear(mask_init[0,0,:,:], opt)
        else:
            mask_init = mask_init[0,0,:,:]
        mask_lerp = db_to_linear(mask_lerp[0,0,:,:], opt)
    
        if opt.mask_ys != None and opt.mask_ye != None:
            re_mask = mask.clone()
            re_mask[:]=0
            re_mask[int(opt.mask_ys / 22050 * 1025):int(opt.mask_ye / 22050 * 1025), mask_start:mask_end] = 1
            mask = re_mask

        mask_init = mask_init * mask
        mask_lerp = mask_lerp * mask
        masked_gt = gt * (1 - mask)
        masked_gt_init = gt * (1 - mask) + mask_init
        masked_gt_lerp = gt * (1 - mask) + mask_lerp
        second = db_to_linear(second_out[0,0,:,:], opt)
        seconded_img = gt * (1 - mask) + second * mask
        # second_out_wholeimg = img * (1 - mask) + second_out * mask
        # seconded_img = db_to_linear(second_out_wholeimg[0,0,:,:], opt)

        img_list = [gt.detach().cpu(), mask.detach().cpu(), mask_init.detach().cpu(), masked_gt_init.detach().cpu(), second.detach().cpu(), seconded_img.detach().cpu(), masked_gt_lerp.detach().cpu()]

        if opt.spec_pow == 2:
            dbpow = 'pow'
        elif opt.spec_pow == 1:
            dbpow = 'amp'
        utils.save_samples_val(sample_folder = save_dir, sample_name = opt.sample_name, img_list = img_list, dbpow=dbpow)

        gt_pad = torch.nn.functional.pad(gt, (0, 3, 0, 1), mode='constant', value=0)
        mask_pad = torch.nn.functional.pad(mask, (0, 3, 0, 1), mode='constant', value=0)
        masked_gt_pad = torch.nn.functional.pad(masked_gt, (0, 3, 0, 1), mode='constant', value=0)
        masked_gt_init_pad = torch.nn.functional.pad(masked_gt_init, (0, 3, 0, 1), mode='constant', value=0)
        masked_gt_lerp_pad = torch.nn.functional.pad(masked_gt_lerp, (0, 3, 0, 1), mode='constant', value=0)
        second_pad = torch.nn.functional.pad(second, (0, 3, 0, 1), mode='constant', value=0)
        
        # seconded_img = torch.pow(seconded_img, 0.5)
        seconded_img_pad = torch.nn.functional.pad(seconded_img, (0, 3, 0, 1), mode='constant', value=0)

        # img_list = [gt_pad.detach().cpu(), mask_pad.detach().cpu(), mask_init.detach().cpu(), masked_gt_lerp_pad.detach().cpu(), second_pad.detach().cpu(), seconded_img_pad.detach().cpu()]
        # utils.save_samples_val(sample_folder = save_dir, sample_name = opt.sample_name, img_list = img_list, dbpow=dbpow)

        original_audio = original_audio.cuda()
        original_audio = original_audio.unsqueeze(0)
        complex_spec = audio_utils.get_spectrogram(original_audio, power=None, return_complex=1, device='cuda')[0]
        phase_spec = torch.angle(complex_spec)
        mag_spec = torch.abs(complex_spec)
        real_spec = mag_spec * torch.cos(phase_spec)
        img_spec = mag_spec * torch.sin(phase_spec)
        cat_spec = torch.cat([real_spec.unsqueeze(-1), img_spec.unsqueeze(-1)], -1)
        re_audio = torch.istft(cat_spec, n_fft=2048, hop_length=512)

        torchaudio.save(os.path.join(save_dir, opt.sample_name + '_regt.wav'), re_audio.detach().cpu(), sample_rate=44100)

        pred_mag_spec = seconded_img_pad ** 0.5
        
        # pred_mag_spec = mag_spec * (1-mask_pad) + mask_pad * pred_mag_spec

        pred_real_spec = pred_mag_spec * torch.cos(phase_spec)
        pred_img_spec = pred_mag_spec * torch.sin(phase_spec)
        pred_cat_spec = torch.cat([pred_real_spec.unsqueeze(-1), pred_img_spec.unsqueeze(-1)], -1)
        pred_re_audio = torch.istft(pred_cat_spec, n_fft=2048, hop_length=512)
        torchaudio.save(os.path.join(save_dir, opt.sample_name + '_repred.wav'), pred_re_audio.detach().cpu(), sample_rate=44100)
        

        # gfl_gt_pad = torch_gflim(gt_pad).unsqueeze(0)
        # gfl_masked_gt_pad = torch_gflim(masked_gt_pad).unsqueeze(0)
        # gfl_masked_gt_lerp_pad = torch_gflim(masked_gt_lerp_pad).unsqueeze(0)
        # gfl_second_pad = torch_gflim(second_pad).unsqueeze(0)
        gfl_second_img_pad = torch_gflim(seconded_img_pad).unsqueeze(0)

        # gfl_gt_pad = custom_gflim(gt_pad.unsqueeze(0), complex_spec, mask_pad)
        # gfl_masked_gt_pad = custom_gflim(masked_gt_pad.unsqueeze(0), complex_spec, mask_pad)
        gfl_masked_gt_lerp_pad = custom_gflim(masked_gt_lerp_pad.unsqueeze(0), complex_spec, mask_pad)
        # gfl_second_pad = custom_gflim(second_pad.unsqueeze(0), complex_spec, mask_pad)
        gfl_second_img_pad_custom = custom_gflim((seconded_img_pad).unsqueeze(0), complex_spec, mask_pad)
        
        torchaudio.save(os.path.join(save_dir, opt.sample_name + '_gt.wav'), audio[0].detach().cpu(), sample_rate=44100)
        # torchaudio.save(os.path.join(save_dir, opt.sample_name + '_gt_gflim.wav'), gfl_gt_pad.detach().cpu(), sample_rate=44100)
        # torchaudio.save(os.path.join(save_dir, opt.sample_name + '_gt_masked_gflim.wav'), gfl_masked_gt_pad.detach().cpu(), sample_rate=44100)
        torchaudio.save(os.path.join(save_dir, opt.sample_name + '_gt_masked_lerp_gflim.wav'), gfl_masked_gt_lerp_pad.detach().cpu(), sample_rate=44100)
        # torchaudio.save(os.path.join(save_dir, opt.sample_name + '_pred_gfli.wav'), gfl_second_pad.detach().cpu(), sample_rate=44100)
        torchaudio.save(os.path.join(save_dir, opt.sample_name + '_pred_gflim.wav'), gfl_second_img_pad.detach().cpu(), sample_rate=44100)
        torchaudio.save(os.path.join(save_dir, opt.sample_name + '_pred_gflim_custom.wav'), gfl_second_img_pad_custom.detach().cpu(), sample_rate=44100)

# python validation.py --load_folder=210808 --load_model=1 --epoched=21 --gpu_ids=10


# python test.py --load_folder=210807_sp --load_model=1 --epoched=15 --gpu_ids=7 --test_audio='ArtistsCard_F_2_아이유 (IU)_밤편지_0034_recon.wav' --audio_start=2 --audio_end=7 --mask_xs=5 --mask_xe=5.2 --latent_channels=48
# python test.py --load_folder=210807 --load_model=3 --epoched=140 --gpu_ids=7 --test_audio='ArtistsCard_F_2_아이유 (IU)_밤편지_0034_recon.wav' --audio_start=2 --audio_end=7 --mask_xs=5 --mask_xe=5.2 --pos_enc=mel

# python test.py --load_folder=210808 --load_model=1 --epoched=50 --gpu_ids=7 --test_audio='ArtistsCard_F_2_아이유 (IU)_밤편지_0034_recon.wav' --audio_start=2 --audio_end=7 --mask_xs=5 --mask_xe=5.2
# python test.py --load_folder=210808 --load_model=1 --epoched=50 --gpu_ids=7 --test_audio='ArtistsCard_F_18_폴킴_모든 날 모든 순간_0004_recon.wav' --audio_start=0 --mask_xs=2.45 --mask_xe=2.55
# python test.py --load_folder=210808 --load_model=1 --epoched=50 --gpu_ids=7 --test_audio='ArtistsCard_F_5_윤하_봄은 있었다_0028_recon.wav' --audio_start=2 --audio_end=7 --mask_xs=4.4 --mask_xe=4.5
# python test.py --load_folder=210808 --load_model=1 --epoched=50 --gpu_ids=7 --test_audio='ArtistsCard_F_5_윤하_봄은 있었다_0028_recon_pred_gflim.wav' --audio_start=0 --audio_end=5 --mask_xs=2.6 --mask_xe=2.7 --data_dir=/data2/personal/jaejun/inpainting/results/210808/1/test_50
# python test.py --load_folder=210808 --load_model=1 --epoched=50 --gpu_ids=7 --test_audio='ArtistsCard_F_5_윤하_봄은 있었다_0028_recon_pred_gflim_pred_gflim.wav' --audio_start=0 --audio_end=5 --mask_xs=3.5 --mask_xe=3.6 --data_dir=/data2/personal/jaejun/inpainting/results/210808/1/test_50
# python test.py --load_folder=210808 --load_model=1 --epoched=50 --gpu_ids=7 --test_audio='ArtistsCard_F_5_윤하_봄은 있었다_0028_recon_pred_gflim_pred_gflim_pred_gflim.wav' --audio_start=0 --audio_end=5 --mask_xs=3.65 --mask_xe=3.75 --data_dir=/data2/personal/jaejun/inpainting/results/210808/1/test_50

# python test.py --load_folder=210808 --load_model=6 --epoched=21 --gpu_ids=7 --test_audio='ArtistsCard_F_2_아이유 (IU)_밤편지_0034_recon.wav' --audio_start=2 --audio_end=7 --mask_xs=5 --mask_xe=5.2 --pos_enc=mel
# python test.py --load_folder=210808 --load_model=6 --epoched=21 --gpu_ids=7 --test_audio='ArtistsCard_F_18_폴킴_모든 날 모든 순간_0004_recon.wav' --audio_start=0 --mask_xs=2.45 --mask_xe=2.55 --pos_enc=mel

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=17 --mask_xs=21.17 --mask_xe=21.37 --pos_enc=cartesian --mask_init=None --scale=2
# python test.py --load_folder=210811_sp --load_model=7 --epoched=45 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=17 --mask_xs=21.17 --mask_xe=21.37 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=2
# python test.py --load_folder=210811 --load_model=2 --epoched=140 --gpu_ids=11 --test_audio='낡은테잎voxdemo.wav' --audio_start=17 --mask_xs=21.17 --mask_xe=21.37 --pos_enc=cartesian --mask_init=lerp --scale=2
# python test.py --load_folder=210808 --load_model=7 --epoched=120 --gpu_ids=12 --test_audio='낡은테잎voxdemo.wav' --audio_start=17 --mask_xs=21.17 --mask_xe=21.37 --pos_enc=cartesian --mask_init=lerp --scale=2



# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=0.5 --mask_xs=4.55 --mask_xe=4.65 --pos_enc=cartesian --mask_init=None --scale=2
# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=0.5 --mask_xs=4.55 --mask_xe=4.65 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=11 --mask_xs=13.3 --mask_xe=13.4 --pos_enc=cartesian --mask_init=randn --scale=2
# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=11 --mask_xs=13.3 --mask_xe=13.4 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=17 --mask_xs=21.2 --mask_xe=21.35 --pos_enc=cartesian --mask_init=None --scale=2
# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=17 --mask_xs=21.2 --mask_xe=21.35 --mask_ys=0 --mask_ye=1500 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=22 --mask_xs=24.3 --mask_xe=24.39 --pos_enc=cartesian --mask_init=None --scale=2
# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=22 --mask_xs=24.3 --mask_xe=24.39 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=32 --mask_xs=34.0 --mask_xe=34.08 --mask_ys=0 --mask_ye=1500 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=42.5 --mask_xs=45.8 --mask_xe=46.0 --mask_ys=0 --mask_ye=2000 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=46.5 --mask_xs=49.02 --mask_xe=49.15 --mask_ys=200 --mask_ye=2500 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=59.5 --mask_xs=60.95 --mask_xe=61.05 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=64 --mask_xs=65.87 --mask_xe=65.93 --mask_ys=0 --mask_ye=1500 --pos_enc=cartesian --mask_init=randn --scale=2
# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=62 --mask_xs=65.85 --mask_xe=65.95 --pos_enc=cartesian --mask_init=None --scale=2

# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=69.5 --mask_xs=70.6 --mask_xe=70.75 --mask_ys=0 --mask_ye=1500 --pos_enc=cartesian --mask_init=None --scale=2



# python test.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --test_audio='src.wav' --audio_start=0 --mask_xs=1.3 --mask_xe=2 --pos_enc=cartesian --mask_init=None --scale=1




# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=1 --test_audio='낡은테잎voxdemo.wav' --audio_start=0.5 --mask_xs=4.55 --mask_xe=4.6 --pos_enc=cartesian --mask_init=None --latent_channels=48 --batch_sized=8 --scale=1 
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=0.5 --mask_xs=4.55 --mask_xe=4.6 --mask_ys=0 --mask_ye=500 --pos_enc=cartesian --mask_init=None --latent_channels=48 --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=0.5 --mask_xs=4.55 --mask_xe=4.6 --mask_ys=0 --mask_ye=500 --pos_enc=cartesian --mask_init=None --latent_channels=48 --batch_sized=8 --scale=1

# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=1 --test_audio='낡은테잎voxdemo.wav' --audio_start=11 --mask_xs=13.3 --mask_xe=13.4 --pos_enc=cartesian --mask_init=None --latent_channels=48 --batch_sized=8 --scale=1 
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=11 --mask_xs=13.3 --mask_xe=13.4 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --mask_init=None --latent_channels=48 --batch_sized=8 --scale=1

# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=19 --mask_xs=21.22 --mask_xe=21.32 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=19 --mask_xs=21.26 --mask_xe=21.37 --mask_ys=200 --mask_ye=1500 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=19 --mask_xs=21.2 --mask_xe=21.4 --mask_ys=200 --mask_ye=1500 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1

# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=24 --mask_xs=24.3 --mask_xe=24.38 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=24 --mask_xs=24.3 --mask_xe=24.38 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1

# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=32 --mask_xs=33.9 --mask_xe=34.09 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=32 --mask_xs=33.9 --mask_xe=34.09 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1

# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=44 --mask_xs=45.9 --mask_xe=46 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=44 --mask_xs=45.9 --mask_xe=46 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=45 --mask_xs=45.91 --mask_xe=45.97 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1


# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=47 --mask_xs=49.05 --mask_xe=49.15 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=47 --mask_xs=49.05 --mask_xe=49.15 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1

# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=59.5 --mask_xs=60.95 --mask_xe=61.05 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=59.5 --mask_xs=60.95 --mask_xe=61.05 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1

# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=64 --mask_xs=65.85 --mask_xe=65.93 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=108 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=64 --mask_xs=65.85 --mask_xe=65.93 --mask_ys=750 --mask_ye=1500 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1


# python test.py --load_folder=210811_sp --load_model=7 --epoched=140 --gpu_ids=9 --test_audio='src.wav' --audio_start=0 --mask_xs=1.4 --mask_xe=1.8 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=140 --gpu_ids=9 --test_audio='src2.wav' --audio_start=0 --mask_xs=1.55 --mask_xe=1.65 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1
# python test.py --load_folder=210811_sp --load_model=7 --epoched=140 --gpu_ids=9 --test_audio='src3.wav' --audio_start=0 --mask_xs=1.55 --mask_xe=1.65 --pos_enc=cartesian --latent_channels=48 --mask_init=None --batch_sized=8 --scale=1




# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=3 --mask_xs=4.45 --mask_xe=4.65 --pos_enc=cartesian --input_length=132300 --image_width=259 --batch_sized=8 --mask_init=None --scale=1
# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=3 --mask_xs=4.45 --mask_xe=4.65 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --batch_sized=8 --input_length=132300 --image_width=259 --mask_init=None --scale=1

# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=11 --mask_xs=13.3 --mask_xe=13.4 --pos_enc=cartesian --input_length=132300 --image_width=259 --batch_sized=8 --mask_init=None --scale=1
# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=11 --mask_xs=13.3 --mask_xe=13.4 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --batch_sized=8 --input_length=132300 --image_width=259 --mask_init=None --scale=1

# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=21 --mask_xs=21.2 --mask_xe=21.4 --pos_enc=cartesian --input_length=132300 --image_width=259 --batch_sized=8 --mask_init=None --scale=1
# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=21 --mask_xs=21.2 --mask_xe=21.4 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --batch_sized=8 --input_length=132300 --image_width=259 --mask_init=None --scale=1

# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=22 --mask_xs=24.3 --mask_xe=24.39 --pos_enc=cartesian --input_length=132300 --image_width=259 --batch_sized=8 --mask_init=None --scale=1
# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=22 --mask_xs=24.3 --mask_xe=24.39 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --batch_sized=8 --input_length=132300 --image_width=259 --mask_init=None --scale=1

# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=32 --mask_xs=34.0 --mask_xe=34.08 --pos_enc=cartesian --input_length=132300 --image_width=259 --batch_sized=8 --mask_init=None --scale=1
# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=32 --mask_xs=34.0 --mask_xe=34.08 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --batch_sized=8 --input_length=132300 --image_width=259 --mask_init=None --scale=1

# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=22 --mask_xs=24.3 --mask_xe=24.39 --pos_enc=cartesian --input_length=132300 --image_width=259 --batch_sized=8 --mask_init=None --scale=1
# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=22 --mask_xs=24.3 --mask_xe=24.39 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --batch_sized=8 --input_length=132300 --image_width=259 --mask_init=None --scale=1

# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=22 --mask_xs=24.3 --mask_xe=24.39 --pos_enc=cartesian --input_length=132300 --image_width=259 --batch_sized=8 --mask_init=None --scale=1
# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=22 --mask_xs=24.3 --mask_xe=24.39 --mask_ys=0 --mask_ye=1000 --pos_enc=cartesian --batch_sized=8 --input_length=132300 --image_width=259 --mask_init=None --scale=1


# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=9 --test_audio='낡은테잎voxdemo.wav' --audio_start=21 --mask_xs=21.2 --mask_xe=21.3 --pos_enc=cartesian --input_length=132300 --image_width=259 --batch_sized=8 --mask_init=None --scale=1
# python test.py --load_folder=210820 --load_model=1 --epoched=200 --gpu_ids=10 --test_audio='낡은테잎voxdemo.wav' --audio_start=21 --mask_xs=21.2 --mask_xe=21.3 --mask_ys=0 --mask_ye=1500 --pos_enc=cartesian --batch_sized=8 --input_length=132300 --image_width=259 --mask_init=None --scale=1

