import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv

import network
import dataset
import librosa

import matplotlib
import matplotlib.pyplot as plt
import sklimage.measure

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
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_perceptualnet():
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    # Pre-trained VGG-16
    vgg16 = torch.load('./vgg16_pretrained.pth')
    load_dict(perceptualnet, vgg16)
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

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)psnr(spec.unsqueeze(0), (spec*(1-mask) + lerp_mask).unsqueeze(0))

        file.write(str(content[i]) + '\n')
    file.close()

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

def save_samples(sample_folder, sample_name, img_list, scaler):
    # Save image one-by-one

    scaler = scaler
    
    gt = img_list[0].numpy()
    mask = img_list[1].numpy()
    masked_gt = img_list[2].numpy()
    first = img_list[3].cpu().numpy()
    firsted_img = img_list[4].numpy()
    second = img_list[5].numpy()
    seconded_img = img_list[6].numpy()

    fig, axes = plt.subplots(2, 4)
    plot_spectrogram(gt, axes[0, 0])
    plot_spectrogram(mask*255, axes[0, 1])
    plot_spectrogram(masked_gt, axes[0, 2])
    plot_spectrogram(first, axes[1, 0])
    plot_spectrogram(firsted_img, axes[1, 1])
    plot_spectrogram(second, axes[1, 2])
    plot_spectrogram(seconded_img, axes[1, 3])
    fig.set_size_inches(24, 12)
    fig.tight_layout()
    # plt.savefig(os.path.join(sample_folder, sample_name + '_.png'))


def psnr(pred, target, pixel_max_cnt = 100):
    mse = torch.mul(target - pred, target - pred)
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

def plot_spectrogram(spec, ax , title=None, ylabel='freq_bin', aspect='auto', xmax=None):
#   ax.set_title(title or 'Spectrogram (db)')
#   ax.set_ylabel(ylabel)
#   ax.set_xlabel('frame')

#   im = ax.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  im = ax.imshow(spec, origin='lower', aspect=aspect)
  if xmax:
    ax.set_xlim((0, xmax))
  return im