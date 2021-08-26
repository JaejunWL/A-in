import os
import time
import datetime
import numpy as np
import cv2
import torch
import torchaudio
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import network
import dataset
import utils
import audio_utils

import wandb

def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    if opt.perceptual != None:
        perceptualnet = utils.create_perceptualnet(opt)

    wandb.watch(generator)
    wandb.watch(discriminator)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        if opt.perceptual != None:
            perceptualnet = nn.DataParallel(perceptualnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if opt.perceptual != None:
            perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if opt.perceptual != None:
            perceptualnet = perceptualnet.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()
    bw_L1Loss = nn.L1Loss(reduction='none')
    bce_loss = torch.nn.BCELoss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    Tensor = torch.cuda.FloatTensor

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # def gradient_penalty_test(real_data):
    #     alpha = torch.rand(real_data.shape[0], 1, 1, 1)
    #     alpha = alpha.expand_as(real_data)
    #     alpha = alpha.cuda()
    #     interpolated = alpha * real_data + (1 - alpha) * fake_data
    #     interpolated = Variable(interpolated, requires_grad=True)
    #     interpolated = interpolated.cuda()
    #     return interpolated

    def gradient_penalty_test(real_data, fake_data, discriminator, mask, mask_start, mask_end):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.cuda()
        prob_interpolated = discriminator(interpolated, mask.data, mask_start.detach(), mask_end.detach())
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                            create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(real_data.shape[0], -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = opt.gp_weight * ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------
    # Define the dataset
    trainset = dataset.InpaintDataset(opt, split='TRAIN')
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size,
                            shuffle = True, num_workers = opt.num_workers, pin_memory = True,
                            drop_last=True, worker_init_fn = lambda _: np.random.seed())
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

        
    torch_gflim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512, power=opt.spec_pow)
    # torch_gflim.cuda()
    # custom_gflim = audio_utils.Custom_GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512, power=opt.spec_pow)
    # custom_gflim.cuda()

    # Initialize start time
    prev_time = time.time()
    steps = 0
    example_images = []
    mag_L1Loss = 0
    phase_L1Loss = 0

    # torch.autograd.set_detect_anomaly(True)

    # Training loop
    for epoch in range(opt.epochs):
        D_losses = []
        GAN_Losses = []
        # first_MaskL1Losses = []
        second_MaskL1Losses = []
        mag_L1Losses = []
        phase_L1Losses = []
        PerceptualLosses = []
        for batch_idx, (audio, img, mask, mask_init, mask_lerp, mask_start, mask_end) in enumerate(dataloader):

            img = img[:,:,:opt.image_height-1,:opt.image_width-3]
            mask = mask[:,:,:opt.image_height-1,:opt.image_width-3]
            mask_init = mask_init[:,:,:opt.image_height-1,:opt.image_width-3]
            mask_lerp = mask_lerp[:,:,:opt.image_height-1,:opt.image_width-3]

            img = img.cuda()
            mask = mask.cuda()
            mask_init = mask_init.cuda()
            mask_lerp = mask_lerp.cuda()
            mask_start = mask_start.cuda()
            mask_end = mask_end.cuda()

            mask_sum = torch.sum(mask, [-3, -2, -1]).detach()
            mask_loss_scaler = opt.image_height * opt.image_width / mask_sum
            ### Train Discriminator
            optimizer_d.zero_grad()

            # Generator output
            second_out = generator(img.data, mask.data, mask_init.data)
            # forward propagation
            second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

            if opt.discriminator_in_channel > 1:
                real_data, fake_data = img, second_out_wholeimg
            else:
                real_data, fake_data = img[:,0:1,:,:], second_out_wholeimg[:,0:1,:,:]

            if opt.gp_weight != None:
                true_scalar = discriminator(real_data.data, mask.data, mask_start.detach(), mask_end.detach())
                fake_scalar = discriminator(fake_data.data, mask.data, mask_start.detach(), mask_end.detach())
                gradient_penalty = gradient_penalty_test(real_data.data, fake_data.data, discriminator, mask.data, mask_start, mask_end)
                loss_D = - torch.mean(true_scalar) + torch.mean(fake_scalar) + gradient_penalty
            elif opt.gp_weight == None:
                fake_scalar = discriminator(fake_data.detach(), mask, mask_start.detach(), mask_end.detach())
                true_scalar = discriminator(real_data, mask, mask_start.detach(), mask_end.detach())
                if opt.gan_type == 'WGAN':
                    loss_D = - torch.mean(true_scalar) + torch.mean(fake_scalar)
                else:
                    real_label = Variable(Tensor(real_data.size(0), 1).fill_(1.0), requires_grad=False)
                    fake_label = Variable(Tensor(real_data.size(0), 1).fill_(0.0), requires_grad=False)
                    real_loss = bce_loss(true_scalar, real_label)
                    fake_loss = bce_loss(fake_scalar, fake_label)
                    loss_D = (real_loss + fake_loss) / 2

            loss_D.backward()
            optimizer_d.step()

            ### Train Generator
            optimizer_g.zero_grad()

            # Mask L1 Loss
            if opt.loss_region == 1:
                second_MaskL1Loss = L1Loss(second_out_wholeimg[:,0:1,:,:], img[:,0:1,:,:])
                if opt.phase == 1:
                    mag_L1Loss = L1Loss(second_out_wholeimg[:,0:1,:,:], img[:,0:1,:,:])
                    phase_L1Loss = L1Loss(second_out_wholeimg[:,1:2,:,:], img[:,1:2,:,:])
            elif opt.loss_region == 2:
                bw_second_MaskL1Loss = torch.mean(bw_L1Loss(second_out_wholeimg[:,0:1,:,:]*mask, img[:,0:1,:,:]*mask), [1,2,3]) * mask_loss_scaler
                second_MaskL1Loss = torch.mean(bw_second_MaskL1Loss)
                if opt.phase == 1:
                    bw_mag_L1Loss = torch.mean(bw_L1Loss(second_out_wholeimg[:,0:1,:,:]*mask, img[:,0:1,:,:]*mask), [1,2,3]) * mask_loss_scaler
                    bw_phase_L1Loss = torch.mean(bw_L1Loss(second_out_wholeimg[:,1:2,:,:]*mask, img[:,1:2,:,:]*mask), [1,2,3]) * mask_loss_scaler
                    mag_L1Loss, phase_L1Loss = torch.mean(bw_mag_L1Loss), torch.mean(bw_phase_L1Loss)
            elif opt.loss_region == 3:
                step_lr = max(0.1, 0.9**(steps/1000))
                loss_1 = L1Loss(second_out_wholeimg[:,0:1,:,:], img[:,0:1,:,:])
                loss_2 = torch.mean(bw_L1Loss(second_out_wholeimg[:,0:1,:,:]*mask, img[:,0:1,:,:]*mask), [1,2,3]) * mask_loss_scaler
                second_MaskL1Loss = step_lr*loss_1 + torch.mean(loss_2)
                if opt.phase == 1:
                    mag_loss_1 = L1Loss(second_out_wholeimg[:,0:1,:,:], img[:,0:1,:,:])
                    mag_loss_2 = torch.mean(bw_L1Loss(second_out_wholeimg[:,0:1,:,:]*mask, img[:,0:1,:,:]*mask), [1,2,3]) * mask_loss_scaler
                    mag_L1Loss = step_lr * mag_loss_1 + torch.mean(mag_loss_2)
                    phase_loss_1 = L1Loss(second_out_wholeimg[:,1:2,:,:], img[:,1:2,:,:])
                    phase_loss_2 = torch.mean(bw_L1Loss(second_out_wholeimg[:,1:2,:,:]*mask, img[:,0:1,:,:]*mask), [1,2,3]) * mask_loss_scaler
                    phase_L1Loss = step_lr * phase_loss_1 + torch.mean(phase_loss_2)

            # GAN Loss
            fake_scalar = discriminator(fake_data, mask, mask_start, mask_end)

            if opt.gan_type == 'WGAN':
                GAN_Loss = - torch.mean(fake_scalar)
            else:
                GAN_Loss = bce_loss(fake_scalar, real_label)

            # Get the deep semantic feature maps, and compute Perceptual Loss
            if opt.perceptual != None:
                mel_22050_fake = utils.make_perceptual_input(fake_data, mask_start.detach(), mask_end.detach(), opt)
                mel_22050_real = utils.make_perceptual_input(real_data.detach(), mask_start.detach(), mask_end.detach(), opt)
                prcptl_input_fake = {'mel':mel_22050_fake[:,0,:,:]}
                prcptl_input_real = {'mel':mel_22050_real[:,0,:,:]}
                featuremaps_fake = perceptualnet(prcptl_input_fake)          # feature maps
                featuremaps_real = perceptualnet(prcptl_input_real)          # feature maps
                
                bw_PerceptualLoss = torch.mean(bw_L1Loss(featuremaps_fake, featuremaps_real), [1,2]) * mask_loss_scaler
                PerceptualLoss = torch.mean(bw_PerceptualLoss)
                # PerceptualLoss = L1Loss(featuremaps_fake, featuremaps_real)

            # Compute losses
            if epoch < 2:
                loss = opt.lambda_l1 * second_MaskL1Loss
            else:
                loss = opt.lambda_l1 * second_MaskL1Loss + opt.lambda_gan * GAN_Loss            
            if opt.perceptual != None:
                loss += opt.lambda_perceptual * PerceptualLoss

            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            D_losses.append(loss_D.detach().cpu())
            GAN_Losses.append(GAN_Loss.detach().cpu())
            second_MaskL1Losses.append(second_MaskL1Loss.detach().cpu())

            if opt.phase == 1:
                mag_L1Losses.append(mag_L1Loss.detach().cpu())
                phase_L1Losses.append(phase_L1Loss.detach().cpu())

            if opt.perceptual != None:
                PerceptualLosses.append(PerceptualLoss.detach().cpu())

            # Print log
            # print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                # ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.item(), second_MaskL1Loss.item()))
            print("\r[Epoch %d/%d] [Batch %d/%d] [second Mask L1 Loss: %.5f]" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), second_MaskL1Loss.item()))
            if opt.perceptual != None:
                print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" %
                    (loss_D.item(), GAN_Loss.item(), PerceptualLoss.item(), time_left))
            else:
                print("\r[D Loss: %.5f] [G Loss: %.5f] time_left: %s \n" %
                    (loss_D.item(), GAN_Loss.item(), time_left))
            
            if opt.phase == 0:
                wandb.log({"D_loss": loss_D, "G_loss": GAN_Loss, "recon loss": second_MaskL1Loss})
            elif opt.phase == 1:
                wandb.log({"D_loss": loss_D, "G_loss": GAN_Loss, "recon loss": second_MaskL1Loss, "mag loss": mag_L1Loss, "phase loss":phase_L1Loss})
            # print("\rtime_left: %s" % (time_left))
            if opt.perceptual != None:
                wandb.log({"Perceptual": PerceptualLoss})

            steps += 1

            if opt.test != None:
                if batch_idx > 5:
                    break

        if opt.phase == 0:
            wandb.log({"epoch": epoch, "Avg_D_loss": torch.mean(torch.tensor(D_losses)), "Avg_G_loss": torch.mean(torch.tensor(GAN_Losses)),
                        "Avg_recon_loss": torch.mean(torch.tensor(second_MaskL1Losses))})
        elif opt.phase == 1:
            wandb.log({"epoch": epoch, "Avg_D_loss": torch.mean(torch.tensor(D_losses)), "Avg_G_loss": torch.mean(torch.tensor(GAN_Losses)),
                        "Avg_recon_loss": torch.mean(torch.tensor(second_MaskL1Losses)), "Avg_mag_loss": torch.mean(torch.tensor(mag_L1Losses)), "Avg_phase_loss": torch.mean(torch.tensor(phase_L1Losses))})
        if opt.perceptual != None:
            wandb.log({"epoch": epoch, "Avg_prcptl_loss": torch.mean(torch.tensor(PerceptualLosses))})

        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        save_model(generator, (epoch + 1), opt)

        ### Sample data every epoch

        # mask = torch.cat((mask, mask, mask), 1)
        if (epoch + 1) % 1 == 0:
            # gt = img[0,0,:,:] * scaler
            # mask = mask[0,0,:,:]
            # mask_init = mask_init[0,0,:,:]
            # masked_gt = gt * (1 - mask) + mask_init
            # masked_gt = masked_gt * scaler
            # # first = first_out[0,0,:,:] * scaler
            # # firsted_img = gt * (1 - mask) + first * mask
            # second = second_out[0,0,:,:] * scaler
            # seconded_img = gt * (1 - mask) + second * mask

            # # img_list = [gt.detach().cpu(), mask.detach().cpu(), mask_init.detach().cpu(), masked_gt.detach().cpu(), first.detach().cpu(), firsted_img.detach().cpu(), second.detach().cpu(), seconded_img.detach().cpu()]
            # img_list = [gt.detach().cpu(), mask.detach().cpu(), mask_init.detach().cpu(), masked_gt.detach().cpu(), second.detach().cpu(), seconded_img.detach().cpu()]

            # # name_list = ['gt_mag', 'mask', 'masked_gt_mag', 'first_out', 'second_out']
            # utils.save_samples(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, dbpow='db')

            gt = utils.db_to_linear(img[0,0,:,:], opt)
            mask = mask[0,0,:,:]
            if opt.mask_init == 'lerp':
                mask_init = utils.db_to_linear(mask_init[0,0,:,:], opt) * mask
            else:
                mask_init = mask_init[0,0,:,:] * mask
            masked_gt = gt * (1 - mask)
            masked_gt_init = gt * (1 - mask) + mask_init
            second = utils.db_to_linear(second_out[0,0,:,:], opt)
            seconded_img = gt * (1 - mask) + second * mask

            img_list = [gt.detach().cpu(), mask.detach().cpu(), mask_init.detach().cpu(), masked_gt_init.detach().cpu(), second.detach().cpu(), seconded_img.detach().cpu()]

            if opt.spec_pow == 2:
                dbpow = 'pow'
            elif opt.spec_pow == 1:
                dbpow = 'amp'
            utils.save_samples(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, dbpow=dbpow)

            gt = gt.detach().cpu()
            mask = mask.detach().cpu()
            mask_init = mask_init.detach().cpu()
            masked_gt = masked_gt.detach().cpu()
            masked_gt_init = masked_gt_init.detach().cpu()
            second = second.detach().cpu()
            seconded_img = seconded_img.detach().cpu()

            gt_pad = torch.nn.functional.pad(gt, (0, 3, 0, 1), mode='constant', value=0)
            mask_pad = torch.nn.functional.pad(mask, (0, 3, 0, 1), mode='constant', value=0)
            masked_gt_pad = torch.nn.functional.pad(masked_gt, (0, 3, 0, 1), mode='constant', value=0)
            masked_gt_init_pad = torch.nn.functional.pad(masked_gt_init, (0, 3, 0, 1), mode='constant', value=0)
            second_pad = torch.nn.functional.pad(second, (0, 3, 0, 1), mode='constant', value=0)
            seconded_img_pad = torch.nn.functional.pad(seconded_img, (0, 3, 0, 1), mode='constant', value=0)

            # audio = audio.cuda()
            # complex_spec = audio_utils.get_spectrogram(audio[0], power=None, return_complex=1, device='cuda')
            
            gfl_gt_pad = torch_gflim(gt_pad).unsqueeze(0)
            gfl_masked_gt_pad = torch_gflim(masked_gt_pad).unsqueeze(0)
            gfl_masked_gt_init_pad = torch_gflim(masked_gt_init_pad).unsqueeze(0)
            gfl_second_pad = torch_gflim(second_pad).unsqueeze(0)
            gfl_second_img_pad = torch_gflim(seconded_img_pad).unsqueeze(0)
            
            # gfl_gt_pad = custom_gflim(gt_pad.unsqueeze(0), complex_spec, mask_pad)
            # gfl_masked_gt_pad = custom_gflim(masked_gt_pad.unsqueeze(0), complex_spec, mask_pad)
            # gfl_masked_gt_init_pad = custom_gflim(masked_gt_init_pad.unsqueeze(0), complex_spec, mask_pad)
            # gfl_second_pad = custom_gflim(second_pad.unsqueeze(0), complex_spec, mask_pad)
            # gfl_second_img_pad = custom_gflim(seconded_img_pad.unsqueeze(0), complex_spec, mask_pad)
            
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_gt.wav'), audio[0].detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_gt_gflim.wav'), gfl_gt_pad.detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_gt_masked_gflim.wav'), gfl_masked_gt_pad.detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_gt_masked_lerp_gflim.wav'), gfl_masked_gt_init_pad.detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_pred_gfli.wav'), gfl_second_pad.detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_pred_gflim.wav'), gfl_second_img_pad.detach().cpu(), sample_rate=44100)