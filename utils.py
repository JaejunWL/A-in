import os
import numpy as np
import cv2
import torch
import torchaudio
import torch.nn as nn
import torchvision as tv

from tony.tony_network import transcriber
from tony.tony_args import make_tony_args

import network
import dataset
import audio_utils

import librosa

import matplotlib
import matplotlib.pyplot as plt
import skimage.measure


# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GatedGenerator(opt)
    print('Generator is created!')
    if opt.load_name:
        # generator = load_dict(generator, opt.load_name)
        generator.load_state_dict(torch.load(opt.load_name))
    else:
        # Init the networks
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    if opt.discriminator == 'patch':
        discriminator = network.PatchDiscriminator(opt)
    elif opt.discriminator == 'jj':
        discriminator = network.jj_Discriminator(opt)
    elif opt.discriminator == 'multi':
        discriminator = network.Multi_Scale_Discriminator(opt)

    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_perceptualnet(opt):
    tony_args = make_tony_args(opt)
    # Get the conv layers and first 4 layers of conformer
    perceptualnet = transcriber.jj_Conformer(tony_args)
    model_dict = perceptualnet.state_dict()

    tony_model_folder = '/data1/singing_inpainting/transcription/results/conformer_small_mel_only/100'
    tony_model_name = 'conformer_small_mel_only_transcriber_100.pt'
    tony_model = torch.load(os.path.join(tony_model_folder, tony_model_name))
    tony_model_dict = tony_model['model']

    pretrained_dict = {k.split('module.')[-1]: v for k, v in tony_model_dict.items() if k.split('module.')[-1] in model_dict}
    # pretrained_dict = {k: v for k, v in tony_model_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    perceptualnet.load_state_dict(model_dict)

    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print('Perceptual network is created!')
    return perceptualnet

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net
    
# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_names(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

# def text_save(content, filename, mode = 'a'):
#     # save a list to a txt
#     # Try to save a list variable in txt file.
#     file = open(filename, mode)psnr(spec.unsqueeze(0), (spec*(1-mask) + lerp_mask).unsqueeze(0))

#         file.write(str(content[i]) + '\n')
#     file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def save_samples(sample_folder, sample_name, img_list, dbpow):
    # Save image one-by-one

    gt = img_list[0].numpy()
    mask = img_list[1].numpy()
    mask_init = img_list[2].numpy()
    masked_gt = img_list[3].numpy()
    second = img_list[4].numpy()
    seconded_img = img_list[5].numpy()

    fig, axes = plt.subplots(2, 3)
    plot_spectrogram(gt, axes[0, 0], dbpow=dbpow)
    plot_spectrogram(mask, axes[0, 1], dbpow=dbpow)
    plot_spectrogram(mask_init, axes[0, 2], dbpow=dbpow)
    plot_spectrogram(masked_gt, axes[1, 0], dbpow=dbpow)
    plot_spectrogram(second, axes[1, 1], dbpow=dbpow)
    plot_spectrogram(seconded_img, axes[1, 2], dbpow=dbpow)

    fig.set_size_inches(24, 12)
    fig.tight_layout()
    plt.savefig(os.path.join(sample_folder, sample_name + '_.png'))

def save_samples_val(sample_folder, sample_name, img_list, dbpow):
    # Save image one-by-one

    gt = img_list[0].numpy()
    mask = img_list[1].numpy()
    mask_init = img_list[2].numpy()
    masked_gt = img_list[3].numpy()
    second = img_list[4].numpy()
    seconded_img = img_list[5].numpy()
    mask_lerp = img_list[6].numpy()

    fig, axes = plt.subplots(2, 3)

    if dbpow == 'db':
        vmin, vmax = gt.min()-10, gt.max()+10
        axes[0, 0].imshow(gt, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, 1].imshow(gt*(1-mask) + mask * vmax, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, 2].imshow(masked_gt, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 0].imshow(seconded_img, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 1].imshow(second, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 2].imshow(mask_lerp, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    elif dbpow == 'pow':
        vmin, vmax = librosa.power_to_db(gt).min()-10, librosa.power_to_db(gt).max()+10
        axes[0, 0].imshow(librosa.power_to_db(gt), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, 1].imshow(librosa.power_to_db(gt*(1-mask)) + mask * vmax, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, 2].imshow(librosa.power_to_db(masked_gt), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 0].imshow(librosa.power_to_db(seconded_img), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 1].imshow(librosa.power_to_db(second), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 2].imshow(librosa.power_to_db(mask_lerp), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    elif dbpow == 'amp':
        vmin, vmax = librosa.amplitude_to_db(gt).min()-10, librosa.amplitude_to_db(gt).max()+10
        axes[0, 0].imshow(librosa.amplitude_to_db(gt), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, 1].imshow(librosa.amplitude_to_db(gt*(1-mask)) + mask * vmax, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, 2].imshow(librosa.amplitude_to_db(masked_gt), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 0].imshow(librosa.amplitude_to_db(seconded_img), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 1].imshow(librosa.amplitude_to_db(second), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 2].imshow(librosa.amplitude_to_db(mask_lerp), origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)


    fig.set_size_inches(24, 12)
    fig.tight_layout()
    plt.savefig(os.path.join(sample_folder, sample_name + '_.png'))

def save_samples2(sample_folder, sample_name, img_list, dbpow):
    # Save image one-by-one

    gt = img_list[0].numpy()
    mask = img_list[1].numpy()
    mask_init = img_list[2].numpy()
    masked_gt = img_list[3].numpy()
    second = img_list[4].numpy()
    seconded_img = img_list[5].numpy()

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(librosa.power_to_db(gt), origin='lower', aspect='auto', cmap='inferno', vmin=-50, vmax=50)
    axes[0, 1].imshow(librosa.power_to_db(gt * (1-mask) + mask), origin='lower', aspect='auto', cmap='inferno', vmin=-50, vmax=50)
    axes[1, 0].imshow(librosa.power_to_db(seconded_img), origin='lower', aspect='auto', cmap='inferno', vmin=-50, vmax=50)
    axes[1, 1].imshow(librosa.power_to_db(gt * (1-mask) + mask_init), origin='lower', aspect='auto', cmap='inferno', vmin=-50, vmax=50)

    fig.set_size_inches(24, 12)
    fig.tight_layout()
    plt.savefig(os.path.join(sample_folder, sample_name + '_.png'))


def psnr(pred, target, pixel_max_cnt = 100):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def mask_psnr(pred, target, mask_idx, pixel_max_cnt = 100):
    start_idx = mask_idx.min()
    end_idx = mask_idx.max()
    pred_mask = pred[...,start_idx:end_idx+1]
    target_mask = target[...,start_idx:end_idx+1]
    mse = torch.mul(target_mask - pred_mask, target_mask - pred_mask)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

def mask_ssim(pred, target, mask_idx):
    start_idx = mask_idx.min()
    end_idx = mask_idx.max()
    pred_mask = pred[...,start_idx:end_idx+1]
    target_mask = target[...,start_idx:end_idx+1]
    pred = pred_mask[0]
    target = target_mask[0]
    pred = pred.clone().data.permute(1, 2, 0).cpu().numpy()
    target = target.clone().data.permute(1, 2, 0).cpu().numpy()
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

def save_spectrogram(spec, sample_folder, sample_img_name, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.amplitude_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
    plt.savefig(os.path.join(sample_folder, sample_img_name))

def plot_spectrogram(spec, ax, title=None, ylabel='freq_bin', aspect='auto', xmax=None, dbpow='db'):
#   ax.set_title(title or 'Spectrogram (db)')
#   ax.set_ylabel(ylabel)
#   ax.set_xlabel('frame')
    if dbpow == 'db':
        im = ax.imshow(spec, origin='lower', aspect=aspect)
    elif dbpow == 'pow':
        im = ax.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    elif dbpow == 'amp':
        im = ax.imshow(librosa.amplitude_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        ax.set_xlim((0, xmax))
    return im


def db_to_linear(db_input, opt):
    if opt.spec_pow == 2:
        linear_output = torchaudio.functional.DB_to_amplitude(db_input, ref=1.0, power=1).cuda() # for db_to_power
    elif opt.spec_pow == 1:
        linear_output = torchaudio.functional.DB_to_amplitude(db_input, ref=1.0, power=0.5).cuda() # for db_to_amplitude
    return linear_output


def make_perceptual_input(fake_data, mask_start, mask_end, opt):
    mel_length = torch.tensor(240).cuda()
    left_length = torch.round(mel_length/2).int()
    mask_region_length = mask_end - mask_start + 1
    center_point = mask_start + torch.round((mask_region_length)/2).int()
    spec_start = torch.max(center_point - left_length, torch.tensor(0).cuda())
    spec_start = torch.min(spec_start, fake_data.shape[-1] - mel_length - torch.tensor(1).cuda())

    # 1 make receptive index matrix (near mask region)
    # index_matrix = torch.zeros([fake_data.shape[0], mel_length], requires_grad=False).cuda()
    # index_matrix += torch.arange(0, mel_length).cuda()
    # index_matrix += spec_start.unsqueeze(-1)
    # index_matrix = index_matrix.unsqueeze(1).unsqueeze(1)
    # index_matrix = index_matrix.expand(fake_data.shape[0], fake_data.shape[1], 128, index_matrix.shape[-1])
    # index_matrix = index_matrix.type(torch.int64)

    # fake_data = db_to_linear(fake_data, opt)
    # fake_data_pad = torch.nn.functional.pad(fake_data, (0, 0, 0, 1), mode='constant', value=0)
    # mel_fake_data = audio_utils.convert_mel_scale(fake_data_pad)
    # log_mel_fake_data = torch.log10(mel_fake_data+1e-7)
    # mel_spec_22050 = torch.gather(log_mel_fake_data, -1, index_matrix)

    # 2
    repeateds = torch.zeros(fake_data.shape[0], fake_data.shape[1], fake_data.shape[2], mel_length)
    for i in range(fake_data.shape[0]):
        repeat_num = torch.ceil(mel_length / mask_region_length[i])
        mask_region = fake_data[i,:,:,mask_start[i]:mask_end[i]+1]
        repeated = mask_region.repeat(1, 1, 1, repeat_num.int())
        repeated = torch.nn.functional.pad(repeated, (0, torch.max(torch.tensor(0), mel_length - repeated.shape[-1])), mode='constant', value=0)
        repeated = repeated[:,:,:,:mel_length]
        repeateds[i] = repeated

    fake_data = db_to_linear(repeateds, opt)
    fake_data_pad = torch.nn.functional.pad(fake_data, (0, 0, 0, 1), mode='constant', value=0)
    mel_fake_data = audio_utils.convert_mel_scale(fake_data_pad)
    log_mel_fake_data = torch.log10(mel_fake_data+1e-7)
    mel_spec_22050 = log_mel_fake_data

    return mel_spec_22050.cuda()


