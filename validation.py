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
    parser.add_argument('--data_dir', type = str, default = '/data1/singing_inpainting/dataset', help = 'dataset directory')
    parser.add_argument('--dataset', type = str, default = ['margs_test', 'nuss_test', 'vocals_test', 'artistscard-eng', 'artistscard', 'bighit', 'changjo', 'supertone'], help = 'datasets to use')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--gpu_ids', type = str, default = "6", help = '')
    parser.add_argument('--epoched', type = int, default = None, help = 'which epoch model you want to use')
    parser.add_argument('--batch_sized', type = int, default = 4, help = 'A batch size of the model you want to load')
    # Dataset parameters
    parser.add_argument('--mask_type', type = str, default = 'time', help = 'mask type')
    parser.add_argument('--mask_init', type = str, default = 'lerp', help = 'mask initialie point')
    parser.add_argument('--image_height', type = int, default = 1025, help = 'height of image')
    parser.add_argument('--image_width', type = int, default = 431, help = 'width of image')
    parser.add_argument('--input_length', type = int, default = 220500, help = 'input length (sample)')
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
    parser.add_argument('--save_interval', type = int, default = 14, help = 'interval length for save png and audio')
    opt = parser.parse_args()

    opt.load_dir = os.path.join('/data2/personal/jaejun/inpainting/results', opt.load_folder, opt.load_model_folder, 'models')
    opt.model_name = 'deepfillv2_WGAN_epoch' + str(opt.epoched) + '_batchsize' + str(opt.batch_sized) + '.pth'
    opt.result_dir = os.path.join('/data2/personal/jaejun/inpainting/results', opt.load_folder, opt.load_model_folder, 'validation_' + str(opt.epoched))
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

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------
    # Define the dataset
    validationset = dataset.InpaintDataset(opt, split='VALID')
    print('The overall number of images equals to %d' % len(validationset))

    # Define the dataloader
    dataloader = DataLoader(validationset, batch_size = 1, pin_memory = True, num_workers=opt.num_workers)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    model = network.GatedGenerator(opt)
    model_dir = os.path.join(opt.load_dir, opt.model_name)
    model.load_state_dict(torch.load(model_dir))
    model.cuda()

    L1Loss = nn.L1Loss()
    bw_L1Loss = nn.L1Loss(reduction='none')

    torch_gflim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512, power=opt.spec_pow)
    torch_gflim.cuda()
    custom_gflim = audio_utils.Custom_GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512, power=opt.spec_pow)
    custom_gflim.cuda()
    if opt.phase == 1:
        custom_gflim_phase = audio_utils.Custom_GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512, power=opt.spec_pow, rand_init='pred')
        custom_gflim_phase.cuda()

    # def db_to_linear(db, ref=1.0, power=1):
        # return 10.0**(db/10.0) * torch.tensor(ref)
    def db_to_linear(db_input, opt):
        if opt.spec_pow == 2:
            linear_output = torchaudio.functional.DB_to_amplitude(db_input, ref=1.0, power=1)
        elif opt.spec_pow == 1:
            linear_output = torchaudio.functional.DB_to_amplitude(db_input, ref=1.0, power=0.5)
        return linear_output

    first_MaskL1Losses = []
    second_MaskL1Losses = []
    psnrs = []
    ssims = []
    with torch.no_grad():
        for batch_idx, (audio, img, mask, mask_init, mask_lerp, mask_start, mask_end) in enumerate(tqdm(dataloader)):
            img = img[:,:,:opt.image_height-1,:opt.image_width-3]
            mask = mask[:,:,:opt.image_height-1,:opt.image_width-3]
            mask_init = mask_init[:,:,:opt.image_height-1,:opt.image_width-3]
            mask_lerp = mask_lerp[:,:,:opt.image_height-1,:opt.image_width-3]

            img = img.cuda()
            mask = mask.cuda()
            mask_init = mask_init.cuda()
            mask_lerp = mask_lerp.cuda()

            mask_sum = torch.sum(mask, [-3, -2, -1]).detach()
            mask_loss_scaler = opt.image_height * opt.image_width / mask_sum

            second_out = model(img, mask, mask_init)
            
            second_out_wholeimg = img * (1 - mask) + second_out * mask

            bw_second_MaskL1Loss = torch.mean(bw_L1Loss(second_out_wholeimg[:,0:1,:,:]*mask, img[:,0:1,:,:]*mask), [1,2,3]) * mask_loss_scaler
            second_MaskL1Loss = torch.mean(bw_second_MaskL1Loss)

            mask_idx = torch.where(mask[...,:]==1)[-1]
            # psnr = utils.mask_psnr(second_out[:,0:1,:,:], img[:,0:1,:,:], mask_idx, pixel_max_cnt=100)
            # ssim = utils.mask_ssim(second_out[:,0:1,:,:], img[:,0:1,:,:], mask_idx)
            
            second_MaskL1Losses.append(second_MaskL1Loss.detach().cpu().numpy())
            # psnrs.append(psnr)
            # ssims.append(ssim)

            if batch_idx % opt.save_interval == 0:
                gt = db_to_linear(img[0,0,:,:], opt)
                mask = mask[0,0,:,:]
                if opt.mask_init == 'lerp':
                    mask_init = db_to_linear(mask_init[0,0,:,:], opt) * mask
                else:
                    mask_init = mask_init[0,0,:,:] * mask
                mask_lerp = db_to_linear(mask_lerp[0,0,:,:], opt) * mask
                masked_gt = gt * (1 - mask)
                masked_gt_init = gt * (1 - mask) + mask_init
                masked_gt_lerp = gt * (1 - mask) + mask_lerp
                second = db_to_linear(second_out[0,0,:,:], opt)
                seconded_img = gt * (1 - mask) + second * mask

                img_list = [gt.detach().cpu(), mask.detach().cpu(), mask_init.detach().cpu(), masked_gt_init.detach().cpu(), second.detach().cpu(), seconded_img.detach().cpu(), masked_gt_lerp.detach().cpu()]

                if opt.spec_pow == 2:
                    dbpow = 'pow'
                elif opt.spec_pow == 1:
                    dbpow = 'amp'
                utils.save_samples_val(sample_folder = save_dir, sample_name = str(batch_idx), img_list = img_list, dbpow=dbpow)

                gt_pad = torch.nn.functional.pad(gt, (0, 3, 0, 1), mode='constant', value=0)
                mask_pad = torch.nn.functional.pad(mask, (0, 3, 0, 1), mode='constant', value=0)
                masked_gt_pad = torch.nn.functional.pad(masked_gt, (0, 3, 0, 1), mode='constant', value=0)
                masked_gt_init_pad = torch.nn.functional.pad(masked_gt_init, (0, 3, 0, 1), mode='constant', value=0)
                masked_gt_lerp_pad = torch.nn.functional.pad(masked_gt_lerp, (0, 3, 0, 1), mode='constant', value=0)
                second_pad = torch.nn.functional.pad(second, (0, 3, 0, 1), mode='constant', value=0)
                seconded_img_pad = torch.nn.functional.pad(seconded_img, (0, 3, 0, 1), mode='constant', value=0)

                audio = audio.cuda()
                complex_spec = audio_utils.get_spectrogram(audio[0], power=None, return_complex=1, device='cuda')
                
                if opt.phase == 1:
                    seconded_mag_img_pad = seconded_img_pad ** (1 / opt.spec_pow)
                    gt_phase = torch.angle(complex_spec)
                    second_phase = second_out[0,1,:,:]
                    second_phase_pad = torch.nn.functional.pad(second_phase, (0, 3, 0, 1), mode='constant', value=0)
                    seconded_img_pad_phase = gt_phase * (1 - mask_pad) + second_phase_pad * mask_pad
                    
                    seconded_img_real = seconded_mag_img_pad * torch.cos(seconded_img_pad_phase)
                    seconded_img_imaginary = seconded_mag_img_pad * torch.sin(seconded_img_pad_phase)
                    seconded_img_cat = torch.cat([seconded_img_real, seconded_img_imaginary], 0)
                    seconded_img_cat = seconded_img_cat.permute(1, 2, 0).contiguous()
                    seconded_img_cat_complex = torch.view_as_complex(seconded_img_cat)

                gfl_gt_pad = torch_gflim(gt_pad).unsqueeze(0)
                gfl_masked_gt_pad = torch_gflim(masked_gt_pad).unsqueeze(0)
                gfl_masked_gt_init_pad = torch_gflim(masked_gt_init_pad).unsqueeze(0)
                gfl_masked_gt_lerp_pad = torch_gflim(masked_gt_lerp_pad).unsqueeze(0)
                gfl_second_pad = torch_gflim(second_pad).unsqueeze(0)
                gfl_second_img_pad = torch_gflim(seconded_img_pad).unsqueeze(0)
                # print(gfl_gt_pad.shape, gfl_masked_gt_pad.shape, gfl_masked_gt_lerp_pad.shape, gfl_second_pad.shape)

                # print(complex_spec_comp.shape, gt_pad.shape, masked_gt_pad.shape, masked_gt_lerp.shape, second_pad.shape, seconded_img_pad.shape)
                
                # gfl_gt_pad = custom_gflim(gt_pad.unsqueeze(0), complex_spec, mask_pad)
                # gfl_masked_gt_pad = custom_gflim(masked_gt_pad.unsqueeze(0), complex_spec, mask_pad)
                # gfl_masked_gt_lerp_pad = custom_gflim(masked_gt_lerp_pad.unsqueeze(0), complex_spec, mask_pad)
                # gfl_second_pad = custom_gflim(second_pad.unsqueeze(0), complex_spec, mask_pad)
                # gfl_second_img_pad = custom_gflim(seconded_img_pad.unsqueeze(0), complex_spec, mask_pad)
                
                if opt.phase == 1:
                    second_pred_phase_istft = torch.istft(seconded_img_cat_complex, n_fft=2048, hop_length=512, win_length=2048)
                    complex_spec_pred = complex_spec * (1-mask_pad) + seconded_img_cat_complex * mask_pad
                    second_pred_phase_gflim = custom_gflim_phase(seconded_img_pad.unsqueeze(0), complex_spec_pred, mask_pad)
                    torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_pred_phase_istft.wav'), second_pred_phase_istft.unsqueeze(0).detach().cpu(), sample_rate=44100)
                    torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_pred_phase_gflim_init.wav'), second_pred_phase_gflim.detach().cpu(), sample_rate=44100)

                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt.wav'), audio[0].detach().cpu(), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_gflim.wav'), gfl_gt_pad.detach().cpu(), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_masked_gflim.wav'), gfl_masked_gt_pad.detach().cpu(), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_masked_init_gflim.wav'), gfl_masked_gt_init_pad.detach().cpu(), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_masked_lerp_gflim.wav'), gfl_masked_gt_lerp_pad.detach().cpu(), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_pred_gfli.wav'), gfl_second_pad.detach().cpu(), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_pred_gflim.wav'), gfl_second_img_pad.detach().cpu(), sample_rate=44100)

        print("Mask L1:", np.mean(second_MaskL1Losses))
        # print("PSNR   :", np.mean(psnrs))
        # print("SSIM   :", np.mean(ssims))

# python validation.py --load_folder=210808 --load_model=1 --epoched=70 --gpu_ids=9
# python validation.py --load_folder=210808 --load_model=5 --epoched=70 --gpu_ids=10 --pos_enc=mel
# python validation.py --load_folder=210808 --load_model=114 --epoched=120 --gpu_ids=11 --pos_enc=cartesian --perceptual=1
# python validation.py --load_folder=210808 --load_model=115 --epoched=120 --gpu_ids=9 --pos_enc=cartesian --perceptual=1

# python validation.py --load_folder=210811 --load_model=1 --epoched=24 --gpu_ids=9 --pos_enc=cartesian
# python validation.py --load_folder=210811 --load_model=2 --epoched=24 --gpu_ids=10 --pos_enc=cartesian
# python validation.py --load_folder=210811 --load_model=3 --epoched=24 --gpu_ids=11 --pos_enc=cartesian
# python validation.py --load_folder=210811 --load_model=4 --epoched=24 --gpu_ids=12 --pos_enc=cartesian

# python validation.py --load_folder=210811 --load_model=2 --epoched=140 --gpu_ids=9 --pos_enc=cartesian
# python validation.py --load_folder=210811 --load_model=5 --epoched=70 --gpu_ids=10 --pos_enc=cartesian
# python validation.py --load_folder=210811 --load_model=6 --epoched=70 --gpu_ids=11 --pos_enc=cartesian
# python validation.py --load_folder=210811 --load_model=7 --epoched=70 --gpu_ids=12 --pos_enc=cartesian --mask_init=randn

# 210819
# python validation.py --load_folder=210811 --load_model=7 --epoched=180 --gpu_ids=9 --pos_enc=cartesian --mask_init=None
# python validation.py --load_folder=210811 --load_model=8 --epoched=120 --gpu_ids=10 --latent_channels=48 --norm=bn --pos_enc=cartesian --batch_sized=8 --mask_init=None
# python validation.py --load_folder=210817 --load_model=1 --epoched=60 --gpu_ids=11 --pos_enc=cartesian --mask_init=None
# python validation.py --load_folder=210811_sp --load_model=7 --epoched=40 --gpu_ids=12 --latent_channels=48 --pos_enc=cartesian --batch_sized=8 --mask_init=None

# python validation.py --load_folder=210811_sp --load_model=7 --epoched=100 --gpu_ids=11 --latent_channels=48 --pos_enc=cartesian --batch_sized=8 --mask_init=None




# python validation.py --load_folder=210826 --load_model=1 --epoched=44 --gpu_ids=9 --pos_enc=cartesian --mask_init=None --mask_type=ctime --batch_size=8
# python validation.py --load_folder=210826 --load_model=2 --epoched=44 --gpu_ids=10 --pos_enc=cartesian --mask_init=None --mask_type=cbtime --batch_size=8
