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
    # perceptualnet = utils.create_perceptualnet()

    wandb.watch(generator)
    wandb.watch(discriminator)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        # perceptualnet = nn.DataParallel(perceptualnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        # perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        # perceptualnet = perceptualnet.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()
    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

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


    def gradient_penalty_test(real_data, fake_data, discriminator, mask):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda()
        interpolated = alpha * real_data + (1 - alpha) * fake_data.detach()
        interpolated = torch.tensor(interpolated, requires_grad=True)
        interpolated = interpolated.cuda()
        prob_interpolated = discriminator(interpolated, mask)
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
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    def db_to_linear(db_input, opt):
        if opt.spec_pow == 2:
            linear_output = torchaudio.functional.DB_to_amplitude(db_input, ref=1.0, power=1)
        elif opt.spec_pow == 1:
            linear_output = torchaudio.functional.DB_to_amplitude(db_input, ref=1.0, power=0.5)
        return linear_output
        
    torch_gflim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512, power=opt.spec_pow)
    torch_gflim.cuda()
    # custom_gflim = audio_utils.Custom_GriffinLim(n_fft=2048, n_iter=60, win_length=2048, hop_length=512, power=opt.spec_pow)
    # custom_gflim.cuda()

    def get_loss_function():
        return nn.BCELoss()

    # Initialize start time
    prev_time = time.time()
    steps = 0
    example_images = []
    mag_L1Loss = 0
    phase_L1Loss = 0
    # Training loop
    for epoch in range(opt.epochs):
        loss_Ds = []
        GAN_Losses = []
        first_MaskL1Losses = []
        second_MaskL1Losses = []
        mag_L1Losses = []
        phase_L1Losses = []
        if opt.discriminator == 'jj':
            reconst_loss = get_loss_function()

        for batch_idx, (audio, img, mask, mask_init) in enumerate(dataloader):
            if opt.discriminator == 'jj':
                real_labels = torch.ones(mask.shape[0], 1).to('cuda')
                fake_labels = torch.zeros(mask.shape[0], 1).to('cuda') 
            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
            scaler = 1
            img = img / scaler
            img = img[:,:,:opt.image_height-1,:opt.image_width-3]
            mask = mask[:,:,:opt.image_height-1,:opt.image_width-3]
            mask_init = mask_init[:,:,:opt.image_height-1,:opt.image_width-3]

            mask_sum = torch.sum(mask)
            img = img.cuda()
            mask = mask.cuda()
            mask_init = mask_init.cuda()

            ### Train Discriminator
            optimizer_d.zero_grad()

            # Generator output
            second_out = generator(img, mask, mask_init)
            # forward propagation
            # first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]


            if opt.pos_enc == None:
                real_data, fake_data = img[:,0:1,:,:], second_out_wholeimg[:,0:1,:,:]
            else:
                pos_emb = img[:,1:2,:,:].clone()
                real_data = img
                fake_data = torch.cat([second_out_wholeimg[:,0:1,:,:], pos_emb], 1)
            if opt.discriminator == 'patch':
                fake_scalar = discriminator(fake_data.detach(), mask)
                true_scalar = discriminator(real_data, mask)
                loss_D = - torch.mean(true_scalar) + torch.mean(fake_scalar)
            elif opt.discriminator == 'jj':
                fake_scalar = discriminator(fake_data.detach(), mask)
                true_scalar = discriminator(real_data, mask)
                loss_D = - torch.mean(true_scalar) + torch.mean(fake_scalar)
            if opt.gp_weight != None:
                gradient_penalty = gradient_penalty_test(real_data, fake_data, discriminator, mask)
                # alpha = torch.rand(real_data.shape[0], 1, 1, 1)
                # alpha = alpha.expand_as(real_data)
                # alpha = alpha.cuda()
                # interpolated = torch.tensor(alpha * real_data + (1 - alpha) * fake_data.detach(), requires_grad=True)
                
                # prob_interpolated = discriminator(interpolated, mask)
                # gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                #                     grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                #                     create_graph=True, retain_graph=True)[0]
                # gradients = gradients.view(opt.batch_size, -1)
                # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                # gradient_penalty = opt.gp_weight * ((gradients_norm - 1) ** 2).mean()
                loss_D = loss_D + gradient_penalty

            loss_D.backward()
            optimizer_d.step()

            ### Train Generator
            optimizer_g.zero_grad()

            # Mask L1 Loss
            if opt.loss_region == 1:
                second_MaskL1Loss = L1Loss(second_out_wholeimg[:,0:1,:,:], img[:,0:1,:,:])
                if opt.phase == 1:
                    mag_L1Loss = L1Loss(second_out_wholeimg[:,0,:,:], img[:,0,:,:])
                    phase_L1Loss = L1Loss(second_out_wholeimg[:,1,:,:], img[:,1,:,:])
            elif opt.loss_region == 2:
                second_MaskL1Loss = L1Loss(second_out_wholeimg*mask, img*mask) / mask_sum * 10250
                if opt.phase == 1:
                    mag_L1Loss = L1Loss(second_out_wholeimg[:,0,:,:]*mask, img[:,0,:,:]*mask)
                    phase_L1Loss = L1Loss(second_out_wholeimg[:,1,:,:]*mask, img[:,1,:,:]*mask)
            elif opt.loss_region == 3:
                step_lr = max(0.1, 0.9**(steps/1000))
                second_MaskL1Loss = step_lr*L1Loss(second_out_wholeimg[:,0:1,:,:], img) + L1Loss(second_out_wholeimg[:,0:1,:,:]*mask, img[:,0:1,:,:]*mask) / mask_sum * 10250
                if opt.phase == 1:
                    mag_L1Loss = step_lr*L1Loss(second_out_wholeimg[:,0,:,:], img[:,0,:,:]) + L1Loss(second_out_wholeimg[:,0,:,:]*mask, img[:,0,:,:]*mask) / mask_sum * 10250
                    phase_L1Loss = step_lr*L1Loss(second_out_wholeimg[:,1,:,:], img[:,1,:,:]) + L1Loss(second_out_wholeimg[:,1,:,:]*mask, img[:,1,:,:]*mask) / mask_sum * 10250
                steps += 1
            
            # GAN Loss
            fake_scalar = discriminator(fake_data, mask)
            # fake_scalar = discriminator((second_out*mask).detach(), mask)

            GAN_Loss = - torch.mean(fake_scalar)
            # GAN_Loss = - torch.mean(fake_scalar) / mask_sum * 10250

            # Get the deep semantic feature maps, and compute Perceptual Loss
            # img_featuremaps = perceptualnet(img)                            # feature maps
            # second_out_wholeimg_featuremaps = perceptualnet(second_out_wholeimg)
            # second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

            # Compute losses
            # loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + \
                # opt.lambda_perceptual * second_PerceptualLoss + opt.lambda_gan * GAN_Loss
            # loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + \
                # + opt.lambda_gan * GAN_Loss
            loss = opt.lambda_l1 * second_MaskL1Loss + opt.lambda_gan * GAN_Loss

            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            loss_Ds.append(loss_D.detach().cpu())
            GAN_Losses.append(GAN_Loss.detach().cpu())
            # first_MaskL1Losses.append(first_MaskL1Loss)
            second_MaskL1Losses.append(second_MaskL1Loss.detach().cpu())

            if opt.phase == 1:
                mag_L1Losses.append(mag_L1Loss.detach().cpu())
                phase_L1Losses.append(phase_L1Loss.detach().cpu())

            # Print log
            # print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                # ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.item(), second_MaskL1Loss.item()))
            print("\r[Epoch %d/%d] [Batch %d/%d] [second Mask L1 Loss: %.5f]" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), second_MaskL1Loss.item()))
            # print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" %
                # (loss_D.item(), GAN_Loss.item(), second_PerceptualLoss.item(), time_left))
            print("\r[D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                (loss_D.item(), GAN_Loss.item(), time_left))
            
            if opt.phase == 0:
                wandb.log({"D_loss": loss_D, "G_loss": GAN_Loss, "recon loss": second_MaskL1Loss, "mag loss": mag_L1Loss, "phase loss":phase_L1Loss})
            elif opt.phase == 1:
                wandb.log({"D_loss": loss_D, "G_loss": GAN_Loss, "recon loss": second_MaskL1Loss, "mag loss": mag_L1Loss, "phase loss":phase_L1Loss})
            # print("\rtime_left: %s" % (time_left))
            # wandb.log({"epoch": epoch, "first_mask loss": first_MaskL1Loss, "second_mask loss": second_MaskL1Loss,})

            # if batch_idx > 5:
                # break

        # wandb.log({"epoch": epoch, "Avg_D_loss": torch.mean(torch.tensor(loss_Ds)), "Avg_G_loss": torch.mean(torch.tensor(GAN_Losses)), "Avg_fm_loss": torch.mean(torch.tensor(first_MaskL1Losses)), "Avg_sm_loss": torch.mean(torch.tensor(second_MaskL1Losses))})
        if opt.phase == 0:
            wandb.log({"epoch": epoch, "Avg_D_loss": torch.mean(torch.tensor(loss_Ds)), "Avg_G_loss": torch.mean(torch.tensor(GAN_Losses)),
                        "Avg_recon_loss": torch.mean(torch.tensor(second_MaskL1Losses))})
        elif opt.phase == 1:
            wandb.log({"epoch": epoch, "Avg_D_loss": torch.mean(torch.tensor(loss_Ds)), "Avg_G_loss": torch.mean(torch.tensor(GAN_Losses)),
                        "Avg_recon_loss": torch.mean(torch.tensor(second_MaskL1Losses)), "Avg_mag_loss": torch.mean(torch.tensor(mag_L1Losses)), "Avg_phase_loss": torch.mean(torch.tensor(phase_L1Losses))})

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

            gt = db_to_linear(img[0,0,:,:] * scaler, opt)
            mask = mask[0,0,:,:]
            mask_init = db_to_linear(mask_init[0,0,:,:], opt) * mask
            masked_gt = gt * (1 - mask)
            masked_gt_lerp = gt * (1 - mask) + mask_init
            # first = db_to_linear(first_out[0,0,:,:] * scaler, ref=1.0, power=1)
            # firsted_img = gt * (1 - mask) + first * mask
            second = db_to_linear(second_out[0,0,:,:] * scaler, opt)
            seconded_img = gt * (1 - mask) + second * mask

            img_list = [gt.detach().cpu(), mask.detach().cpu(), mask_init.detach().cpu(), masked_gt_lerp.detach().cpu(), second.detach().cpu(), seconded_img.detach().cpu()]

            if opt.spec_pow == 2:
                dbpow = 'pow'
            elif opt.spec_pow == 1:
                dbpow = 'amp'
            utils.save_samples(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, dbpow=dbpow)

            gt_pad = torch.nn.functional.pad(gt, (0, 3, 0, 1), mode='constant', value=0)
            mask_pad = torch.nn.functional.pad(mask, (0, 3, 0, 1), mode='constant', value=0)
            masked_gt_pad = torch.nn.functional.pad(masked_gt, (0, 3, 0, 1), mode='constant', value=0)
            masked_gt_lerp_pad = torch.nn.functional.pad(masked_gt_lerp, (0, 3, 0, 1), mode='constant', value=0)
            second_pad = torch.nn.functional.pad(second, (0, 3, 0, 1), mode='constant', value=0)
            seconded_img_pad = torch.nn.functional.pad(seconded_img, (0, 3, 0, 1), mode='constant', value=0)

            audio = audio.cuda()
            complex_spec = audio_utils.get_spectrogram(audio[0], power=None, return_complex=1, device='cuda')
            
            gfl_gt_pad = torch_gflim(gt_pad).unsqueeze(0)
            gfl_masked_gt_pad = torch_gflim(masked_gt_pad).unsqueeze(0)
            gfl_masked_gt_lerp_pad = torch_gflim(masked_gt_lerp_pad).unsqueeze(0)
            gfl_second_pad = torch_gflim(second_pad).unsqueeze(0)
            gfl_second_img_pad = torch_gflim(seconded_img_pad).unsqueeze(0)
            
            # gfl_gt_pad = custom_gflim(gt_pad.unsqueeze(0), complex_spec, mask_pad)
            # gfl_masked_gt_pad = custom_gflim(masked_gt_pad.unsqueeze(0), complex_spec, mask_pad)
            # gfl_masked_gt_lerp_pad = custom_gflim(masked_gt_lerp_pad.unsqueeze(0), complex_spec, mask_pad)
            # gfl_second_pad = custom_gflim(second_pad.unsqueeze(0), complex_spec, mask_pad)
            # gfl_second_img_pad = custom_gflim(seconded_img_pad.unsqueeze(0), complex_spec, mask_pad)
            
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_gt.wav'), audio[0].detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_gt_gflim.wav'), gfl_gt_pad.detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_gt_masked_gflim.wav'), gfl_masked_gt_pad.detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_gt_masked_lerp_gflim.wav'), gfl_masked_gt_lerp_pad.detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_pred_gfli.wav'), gfl_second_pad.detach().cpu(), sample_rate=44100)
            torchaudio.save(os.path.join(sample_folder, 'epoch' + str(epoch+1) + '_pred_gflim.wav'), gfl_second_img_pad.detach().cpu(), sample_rate=44100)
            


# def LSGAN_trainer(opt):
#     # ----------------------------------------
#     #      Initialize training parameters
#     # ----------------------------------------

#     # cudnn benchmark accelerates the network
#     cudnn.benchmark = opt.cudnn_benchmark

#     # configurations
#     save_folder = opt.save_path
#     sample_folder = opt.sample_path
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     if not os.path.exists(sample_folder):
#         os.makedirs(sample_folder)

#     # Build networks
#     generator = utils.create_generator(opt)
#     discriminator = utils.create_discriminator(opt)
#     # perceptualnet = utils.create_perceptualnet()

#     # To device
#     if opt.multi_gpu == True:
#         generator = nn.DataParallel(generator)
#         discriminator = nn.DataParallel(discriminator)
#         # perceptualnet = nn.DataParallel(perceptualnet)
#         generator = generator.cuda()
#         discriminator = discriminator.cuda()
#         # perceptualnet = perceptualnet.cuda()
#     else:
#         generator = generator.cuda()
#         discriminator = discriminator.cuda()
#         # perceptualnet = perceptualnet.cuda()

#     # Loss functions
#     L1Loss = nn.L1Loss()
#     MSELoss = nn.MSELoss()

#     # Optimizers
#     optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
#     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

#     # Learning rate decrease
#     def adjust_learning_rate(lr_in, optimizer, epoch, opt):
#         """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
#         lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
    
#     # Save the model if pre_train == True
#     def save_model(net, epoch, opt):
#         """Save the model at "checkpoint_interval" and its multiple"""
#         model_name = 'deepfillv2_LSGAN_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
#         model_name = os.path.join(save_folder, model_name)
#         if opt.multi_gpu == True:
#             if epoch % opt.checkpoint_interval == 0:
#                 torch.save(net.module.state_dict(), model_name)
#                 print('The trained model is successfully saved at epoch %d' % (epoch))
#         else:
#             if epoch % opt.checkpoint_interval == 0:
#                 torch.save(net.state_dict(), model_name)
#                 print('The trained model is successfully saved at epoch %d' % (epoch))
    
#     # ----------------------------------------
#     #       Initialize training dataset
#     # ----------------------------------------

#     # Define the dataset
#     trainset = dataset.InpaintDataset(opt, split='TRAIN')

#     print('The overall number of images equals to %d' % len(trainset))

#     # Define the dataloader
#     dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True, worker_init_fn = lambda _: np.random.seed())
    
#     # ----------------------------------------
#     #            Training and Testing
#     # ----------------------------------------

#     # Initialize start time
#     prev_time = time.time()
    
#     # Tensor type
#     Tensor = torch.cuda.FloatTensor

#     # Training loop
#     for epoch in range(opt.epochs):
#         for batch_idx, (img, mask) in enumerate(dataloader):

#             # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
#             img = img / 300
#             img = img[:,:,:512,:428]
#             img = img.cuda()
#             mask = mask[:,:,:512,:428]
#             mask = mask.cuda()

#             # LSGAN vectors
#             valid = Tensor(np.ones((img.shape[0], 1, 8, 8)))
#             fake = Tensor(np.zeros((img.shape[0], 1, 8, 8)))

#             ### Train Discriminator
#             optimizer_d.zero_grad()

#             # Generator output
#             first_out, second_out = generator(img, mask)

#             # forward propagation
#             first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
#             second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

#             # Fake samples
#             fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
#             # True samples
#             true_scalar = discriminator(img, mask)
            
#             # Overall Loss and optimize
#             loss_fake = MSELoss(fake_scalar, fake)
#             loss_true = MSELoss(true_scalar, valid)
#             # Overall Loss and optimize
#             loss_D = 0.5 * (loss_fake + loss_true)
#             loss_D.backward()
#             optimizer_d.step()

#             ### Train Generator
#             optimizer_g.zero_grad()

#             # Mask L1 Loss
#             first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
#             second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
            
#             # GAN Loss
#             fake_scalar = discriminator(second_out_wholeimg, mask)
#             GAN_Loss = MSELoss(fake_scalar, valid)

#             # Get the deep semantic feature maps, and compute Perceptual Loss
#             # img_featuremaps = perceptualnet(img)                            # feature maps
#             # second_out_wholeimg_featuremaps = perceptualnet(second_out_wholeimg)
#             # second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

#             # Compute losses
#             # loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + \
#                 # opt.lambda_perceptual * second_PerceptualLoss + opt.lambda_gan * GAN_Loss
#             loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + \
#                 + opt.lambda_gan * GAN_Loss
#             loss.backward()
#             optimizer_g.step()

#             # Determine approximate time left
#             batches_done = epoch * len(dataloader) + batch_idx
#             batches_left = opt.epochs * len(dataloader) - batches_done
#             time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
#             prev_time = time.time()

#             # Print log
#             print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
#                 ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.item(), second_MaskL1Loss.item()))
#             # print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" %
#                 # (loss_D.item(), GAN_Loss.item(), second_PerceptualLoss.item(), time_left))
#             print("\r[D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
#                 (loss_D.item(), GAN_Loss.item(), time_left))

#         # Learning rate decrease
#         adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
#         adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

#         # Save the model
#         save_model(generator, (epoch + 1), opt)

#         ### Sample data every epoch
#         masked_img = img * (1 - mask) + mask
#         mask = torch.cat((mask, mask, mask), 1)
#         if (epoch + 1) % 1 == 0:
#             img_list = [img, mask, masked_img, first_out, second_out]
#             name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
#             utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
