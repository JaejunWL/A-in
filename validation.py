import os
import numpy as np
import librosa 

import torch
import dataset
import easydict
import argparse
import network
import utils
import torchaudio

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import IPython.display as ipd


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--gpu_ids', type = str, default = "6", help = '')
    parser.add_argument('--data_dir', type = str, default = '/data1/singing_inpainting/dataset', help = 'dataset directory')
    parser.add_argument('--result_dir', type = str, default = '/data2/personal/jaejun/inpainting/results/210719/1/', help = 'results directory')
    parser.add_argument('--model_name', type = str, default = 'models/deepfillv2_WGAN_epoch115_batchsize4.pth', help = 'results directory')
    # Dataset parameters
    parser.add_argument('--mask_type', type = str, default = 'time', help = 'mask type')
    parser.add_argument('--image_height', type = int, default = 1024, help = 'height of image')
    parser.add_argument('--image_width', type = int, default = 428, help = 'width of image')
    parser.add_argument('--input_length', type = int, default = 220500, help = 'input length (sample)')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 2, help = 'input real&complex spec + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 1, help = 'output real&complex spec')
    parser.add_argument('--latent_channels', type = int, default = 32, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    # Other parameters
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------
    # Define the dataset
    testset = dataset.InpaintDataset(opt, split='VALID')
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # Validation results save directory
    validation_idx = opt.model_name.split('_')[2]
    validation_dir = os.path.join(opt.result_dir, 'validation_' + validation_idx)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    save_dir = os.path.join(validation_dir)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    model = network.GatedGenerator(opt)
    model_dir = os.path.join(opt.result_dir, opt.model_name)
    model.load_state_dict(torch.load(model_dir))
    model.cuda()

    torch_gflim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512)

    def db_to_power(db, ref=1.0):
        return 10.0**(db/10.0 + torch.log10(torch.tensor(ref)))

    for batch_idx, (audio, img, mask) in enumerate(dataloader):
        if batch_idx > 50:
            stop
        scaler = 1
        img = img / scaler
        img = img[:,:,:opt.image_height,:opt.image_width]
        img = img.cuda()
        
        mask = mask[:,:,:opt.image_height,:opt.image_width]
        mask = mask.cuda()

        print(img.shape, mask.shape)
        first_out, second_out = model(img.cuda(), mask.cuda())
        # first_out = first_out.detach().cpu()
        # second_out = second_out.detach().cpu()
        
        img = img * scaler
        first_out = first_out * scaler
        second_out = second_out * scaler

        first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
        second_out_wholeimg = img * (1 - mask) + second_out * mask

        gt = img[0,0,:,:]
        mask = mask[0,0,:,:]
        masked_gt = gt * (1 - mask) + mask
        first = first_out[0,0,:,:]
        firsted_img = gt * (1 - mask) + first * mask
        second = second_out[0,0,:,:]
        seconded_img = gt * (1 - mask) + second * mask    

        img_list = [gt, mask, masked_gt, first, firsted_img, second, seconded_img]
        utils.save_samples(sample_folder = save_dir, sample_name = str(batch_idx), img_list = img_list, scaler = scaler)

        gt_pad = torch.nn.functional.pad(gt, (0, 3, 0, 1), mode='constant', value=0)
        masked_gt_pad = torch.nn.functional.pad(masked_gt.detach().cpu(), (0, 3, 0, 1), mode='constant', value=0)
        second_pad = torch.nn.functional.pad(second, (0, 3, 0, 1), mode='constant', value=0)
        seconded_img_pad = torch.nn.functional.pad(seconded_img, (0, 3, 0, 1), mode='constant', value=0)

        # gt = gt.detach().cpu()
        # masked_gt = gt * (1 - mask.detach().cpu())
        # second = second.detach().cpu()
        # seconded_img = seconded_img.detach().cpu()
        
        gfl_gt_pad = torch_gflim(gt_pad)
        gfl_masked_gt_pad = torch_gflim(relu(masked_gt_pad))
        gfl_second_pad = torch_gflim(relu(second_pad))
        gfl_second_img_pad = torch_gflim(relu(seconded_img_pad))
        
        torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt.wav'), audio[0], sample_rate=44100)
        torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_gflim.wav'), gfl_gt_pad.unsqueeze(0), sample_rate=44100)
        torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_gt_masked_gflim.wav'), gfl_masked_gt_pad.unsqueeze(0), sample_rate=44100)
        torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_pred_gfli.wav'), gfl_second_pad.unsqueeze(0), sample_rate=44100)
        torchaudio.save(os.path.join(save_dir, str(batch_idx) + '_pred_gflim.wav'), gfl_second_img_pad.unsqueeze(0), sample_rate=44100)

    for batch_idx, (img, mask) in enumerate(dataloader):

        # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
        img = img.cuda()
        mask = mask.cuda()

        # Generator output
        masked_img = img * (1 - mask)
        fake1, fake2 = model(masked_img, mask)

        # forward propagation
        fusion_fake1 = img * (1 - mask) + fake1 * mask                      # in range [-1, 1]
        fusion_fake2 = img * (1 - mask) + fake2 * mask                      # in range [-1, 1]

        # convert to visible image format
        img = img.cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
        img = (img + 1) * 128
        img = img.astype(np.uint8)
        fusion_fake1 = fusion_fake1.detach().cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
        fusion_fake1 = (fusion_fake1 + 1) * 128
        fusion_fake1 = fusion_fake1.astype(np.uint8)
        fusion_fake2 = fusion_fake2.detach().cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
        fusion_fake2 = (fusion_fake2 + 1) * 128
        fusion_fake2 = fusion_fake2.astype(np.uint8)

        # show
        show_img = np.concatenate((img, fusion_fake1, fusion_fake2), axis = 1)
        r, g, b = cv2.split(show_img)
        show_img = cv2.merge([b, g, r])
        cv2.imshow('comparison.jpg', show_img)
        cv2.imwrite('result_%d.jpg' % batch_idx, show_img)
