import os
import numpy as np
import librosa 

import dataset
import easydict
import argparse
import network
import utils

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
    # parser.add_argument('--save_path', type = str, default = '/data2/personal/jaejun/inpainting/results/210723/1/models', help = 'saving path that is a folder')
    # parser.add_argument('--sample_path', type = str, default = '/data2/personal/jaejun/inpainting/results/210723/1/samples', help = 'training samples path that is a folder')
    parser.add_argument('--data_dir', type = str, default = '/data1/singing_inpainting/dataset', help = 'dataset directory')
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
    # Network parameters
    parser.add_argument('--stage_num', type = int, default = 2, help = 'two stage method or just only stage')
    parser.add_argument('--in_channels', type = int, default = 2, help = 'input mag + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 1, help = 'output mag')
    parser.add_argument('--latent_channels', type = int, default = 32, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    # Other parameters
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--save_interval', type = int, default = 7, help = 'interval length for save png and audio')
    opt = parser.parse_args()

    opt.load_dir = os.path.join('/data2/personal/jaejun/inpainting/results', opt.load_folder, opt.load_model_folder, 'models')
    opt.model_name = 'deepfillv2_WGAN_epoch' + str(opt.epoched) + '_batchsize' + str(opt.batch_sized) + '.pth'
    opt.result_dir = os.path.join('/data2/personal/jaejun/inpainting/results', opt.load_folder, opt.load_model_folder, 'validation_' + str(opt.epoched))
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
    testset = dataset.InpaintDataset(opt, split='VALID')
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = 1, pin_memory = True, num_workers=opt.num_workers)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    model = network.GatedGenerator(opt)
    model_dir = os.path.join(opt.load_dir, opt.model_name)
    model.load_state_dict(torch.load(model_dir))
    model.cuda()

    L1Loss = nn.L1Loss()
    relu = nn.ReLU()
    torch_gflim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512)

    def db_to_power(db, ref=1.0):
        return 10.0**(db/10.0 + torch.log10(torch.tensor(ref)))

    first_MaskL1Losses = []
    second_MaskL1Losses = []
    psnrs = []
    ssims = []
    with torch.no_grad():
        for batch_idx, (audio, img, mask, mask_init) in enumerate(tqdm(dataloader)):
            scaler = 1
            img = img / scaler
            img = img[:,:,:opt.image_height-1,:opt.image_width-3]
            mask = mask[:,:,:opt.image_height-1,:opt.image_width-3]
            mask_init = mask_init[:,:,:opt.image_height-1,:opt.image_width-3]

            img = img.cuda()
            mask = mask.cuda()
            mask_init = mask_init.cuda()

            first_out, second_out = model(img, mask, mask_init)
            # first_out = first_out.detach().cpu()
            # second_out = second_out.detach().cpu()
            
            img = img * scaler
            first_out = first_out * scaler
            second_out = second_out * scaler

            first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask
            first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
            second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
            psnr = utils.psnr(second_out_wholeimg, img)
            ssim = utils.ssim(second_out_wholeimg, img)
            
            first_MaskL1Losses.append(first_MaskL1Loss)
            second_MaskL1Losses.append(second_MaskL1Loss)
            psnrs.append(psnrs)
            ssims.append(ssim)

            gt = img[0,0,:,:] * scaler
            gt = gt.detach().cpu()
            mask = mask[0,0,:,:]
            mask = mask.detach().cpu()
            mask_init = mask_init[0,0,:,:]
            mask_init = mask_init.detach().cpu()
            masked_gt = gt * (1 - mask) + mask
            masked_gt = masked_gt.detach().cpu() * scaler
            masked_gt_lerp = gt * (1 - mask) + mask_init
            masked_gt_lerp = masked_gt_lerp.detach().cpu() * scaler        
            first = first_out[0,0,:,:] * scaler
            first = first.detach().cpu()
            firsted_img = gt * (1 - mask) + first * mask
            firsted_img = firsted_img.detach().cpu()
            second = second_out[0,0,:,:] * scaler
            second = second.detach().cpu()
            seconded_img = gt * (1 - mask) + second * mask
            seconded_img = seconded_img.detach().cpu()


            img_list = [gt, mask, mask_init, masked_gt_lerp, first, firsted_img, second, seconded_img]

            if batch_idx % opt.save_interval == 0:
                utils.save_samples(sample_folder = save_dir, sample_name = str(batch_idx), img_list = img_list, scaler = scaler)

                gt_pad = torch.nn.functional.pad(gt, (0, 3, 0, 1), mode='constant', value=0)
                masked_gt_pad = torch.nn.functional.pad(masked_gt.detach().cpu(), (0, 3, 0, 1), mode='constant', value=0)
                masked_gt_lerp_pad = torch.nn.functional.pad(masked_gt.detach().cpu(), (0, 3, 0, 1), mode='constant', value=0)
                second_pad = torch.nn.functional.pad(second, (0, 3, 0, 1), mode='constant', value=0)
                seconded_img_pad = torch.nn.functional.pad(seconded_img, (0, 3, 0, 1), mode='constant', value=0)

                gfl_gt_pad = torch_gflim(db_to_power(gt_pad))
                gfl_masked_gt_pad = torch_gflim(relu(db_to_power(masked_gt_pad)))
                gfl_masked_gt_lerp_pad = torch_gflim(relu(db_to_power(masked_gt_lerp_pad)))
                gfl_second_pad = torch_gflim(relu(db_to_power(second_pad)))
                gfl_second_img_pad = torch_gflim(relu(db_to_power(seconded_img_pad)))
                
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt.wav'), audio[0], sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_gflim.wav'), gfl_gt_pad.unsqueeze(0), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_masked_gflim.wav'), gfl_masked_gt_pad.unsqueeze(0), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_masked_lerp_gflim.wav'), gfl_masked_gt_lerp_pad.unsqueeze(0), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_pred_gfli.wav'), gfl_second_pad.unsqueeze(0), sample_rate=44100)
                torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_pred_gflim.wav'), gfl_second_img_pad.unsqueeze(0), sample_rate=44100)
        print("L1 loss:", np.mean(second_MaskL1Losses))
        print("PSNR   :", np.mean(psnrs))
        print("SSIM   :", np.mean(ssims))