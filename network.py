import torch
import torch.nn as nn
import torch.nn.init as init

from network_module import *

def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)

#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none'),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # decoder
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad_type, activation = 'none', norm = 'none')
        )
        self.refinement = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none'),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # decoder
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad_type, activation = 'none', norm = 'none')
        )
        # self.refinement = nn.Sequential(
        #     # encoder
        #     GatedConv2d(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none'),
        #     GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     # Bottleneck
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     # decoder
        #     TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        #     GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad_type, activation = 'none', norm = 'none')
        # )

    def forward(self, img, mask, mask_init):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # Coarse
        # print(img.shape, mask.shape)
        
        if self.opt.stage_num == 2:
            first_masked_img = img * (1 - mask) + mask_init
            first_in = torch.cat((first_masked_img, mask), 1)       # in: [B, 4, H, W]
            first_out = self.coarse(first_in)                       # out: [B, 3, H, W]
            # Refinement
            second_masked_img = img * (1 - mask) + first_out * mask
            second_in = torch.cat((second_masked_img, mask), 1)     # in: [B, 4, H, W]
            second_out = self.refinement(second_in)                 # out: [B, 3, H, W]
            return first_out, second_out
        elif self.opt.stage_num == 1:
            # Refinement
            masked_img = img * (1 - mask) + mask_init
            _in = torch.cat((masked_img, mask), 1)
            _out = self.refinement(_in)
            return _out


#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.discriminator_in_channel, opt.msd_latent, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none', sn = False)
        self.block2 = Conv2dLayer(opt.msd_latent, opt.msd_latent * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = False)
        self.block3 = Conv2dLayer(opt.msd_latent * 2, opt.msd_latent * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = False)
        self.block4 = Conv2dLayer(opt.msd_latent * 4, opt.msd_latent * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = False)
        self.block5 = Conv2dLayer(opt.msd_latent * 4, opt.msd_latent * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = False)
        self.block6 = Conv2dLayer(opt.msd_latent * 4, 1, 4, 2, 1, pad_type = opt.pad_type, activation = 'none', norm = 'none', sn = False)

    def forward(self, img, mask, mask_start, mask_end):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        # x = torch.cat((img, mask), 1)
        x = img
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x

class jj_Discriminator(nn.Module):
    def __init__(self, opt):
        super(jj_Discriminator, self).__init__()
        self.n_layers = 5
        if opt.gan_type == 'WGAN':
            self.use_sigmoid = False
        else:
            self.use_sigmoid = True
        use_bias = 1
        norm_layer = nn.InstanceNorm2d
        self.relu = nn.LeakyReLU(0.2, True)

        self.conv1 = nn.Conv2d(2, opt.latent_channels*2, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=use_bias)
        self.bn1 = norm_layer(opt.latent_channels*2)
        nf_mult = 1
        for n in range(1, self.n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.add_module('conv2_' + str(n), nn.Conv2d(opt.latent_channels*2 * nf_mult_prev, opt.latent_channels*2 * nf_mult,
                          kernel_size=(3, 3), stride=2, padding=1, bias=use_bias))
            self.add_module('norm_' + str(n), norm_layer(opt.latent_channels*2 * nf_mult))
        nf_mult_prev = nf_mult
        nf_mult = min(2**self.n_layers, 8)

        self.conv3 = nn.Conv2d(opt.latent_channels*2 * nf_mult_prev, opt.latent_channels*2 * nf_mult,
                      kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.norm3 = norm_layer(opt.latent_channels*2 * nf_mult)
        self.conv4 = nn.Conv2d(opt.latent_channels*2 * nf_mult, 1,
                      kernel_size=3, stride=1, padding=1, bias=use_bias)
        if self.use_sigmoid:
            self.linear1 = nn.Linear(896, 1024)
            self.linear2 = nn.Linear(1024, 1)
            self.sig = nn.Sigmoid()

    def forward(self, input, mask):
        batch_size = input.shape[0]
        input_cat = torch.cat([input, mask], 1)
        net = self.conv1(input_cat)
        netn = self.relu(self.bn1(net))
        for n in range(1, self.n_layers):
            netn = self._modules['conv2_' + str(n)](netn)
            netn = self._modules['norm_' + str(n)](netn)
            netn = self.relu(netn)
        net = self.conv3(netn)
        net = self.norm3(net)
        net = self.relu(net)
        net = self.conv4(net)
        net = net.view(batch_size, -1)
        if self.use_sigmoid:
            net = self.linear1(net)
            net = self.linear2(net)
            net = self.sig(net)
        return net

# # ----------------------------------------
# #            Perceptual Network
# # ----------------------------------------
# # VGG-16 conv4_3 features
# class PerceptualNet(nn.Module):
#     def __init__(self):
#         super(PerceptualNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.ReLU(inplace = True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.ReLU(inplace = True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(128, 256, 3, 1, 1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(256, 512, 3, 1, 1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(512, 512, 3, 1, 1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(512, 512, 3, 1, 1)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         return x


class Scale_Discriminator(nn.Module):
    def __init__(self, opt):
        super(Scale_Discriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.discriminator_in_channel, opt.msd_latent, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.msd_latent, opt.msd_latent * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(opt.msd_latent * 2, opt.msd_latent * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(opt.msd_latent * 4, opt.msd_latent * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block5 = Conv2dLayer(opt.msd_latent * 4, opt.msd_latent * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block6 = Conv2dLayer(opt.msd_latent * 4, 1, 4, 2, 1, pad_type = opt.pad_type, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = img
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)  
        return x

class Scale_Discriminator_small(nn.Module):
    def __init__(self, opt):
        super(Scale_Discriminator_small, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.discriminator_in_channel, opt.msd_latent, (7,4), 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.msd_latent, opt.msd_latent * 2, 4, (2,1), 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(opt.msd_latent * 2, opt.msd_latent * 4, 4, (2,2), 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(opt.msd_latent * 4, opt.msd_latent * 4, 4, (2,2), 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block5 = Conv2dLayer(opt.msd_latent * 4, opt.msd_latent * 4, 4, (2,2), 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block6 = Conv2dLayer(opt.msd_latent * 4, 1, 4, 2, 1, pad_type = opt.pad_type, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = img
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x

class Multi_Scale_Discriminator(nn.Module):
    def __init__(self, opt):
        super(Multi_Scale_Discriminator, self).__init__()
        self.frame_lengths = [16, 32, 64, 128]
        self.scale0_discriminator = Scale_Discriminator_small(opt)
        self.scale1_discriminator = Scale_Discriminator(opt)
        self.scale2_discriminator = Scale_Discriminator(opt)
        self.scale3_discriminator = Scale_Discriminator(opt)
        self.opt = opt

    def make_scale_input(self, frame_length, mask_start, mask_end, max_time_index):
        spec_end_start = mask_start + 0.5 * frame_length
        spec_end_end = mask_end + 0.5 * frame_length
        spec_end = (torch.rand(self.batch_size).cuda() * (spec_end_end - spec_end_start + 1) + spec_end_start).int()        
        spec_end = torch.min(spec_end, torch.tensor(max_time_index).cuda())
        spec_start = spec_end - frame_length
        spec_start = torch.max(spec_start, torch.zeros(self.batch_size).cuda())
        return spec_start.cuda()
    
    def make_gather_input(self, img, spec_start, frame_length):
        index_matrix = torch.zeros([self.batch_size, frame_length], requires_grad=False).cuda()
        index_matrix += torch.arange(0, frame_length).cuda()
        index_matrix += spec_start.unsqueeze(-1)
        index_matrix = index_matrix.unsqueeze(1).unsqueeze(1)
        index_matrix = index_matrix.expand(self.batch_size, img.shape[1], img.shape[2], index_matrix.shape[-1])
        index_matrix = index_matrix.type(torch.int64)
        gathered_img = torch.gather(img, -1, index_matrix)
        return gathered_img

    def forward(self, img, mask, mask_start, mask_end):
        self.batch_size = img.shape[0]
        scale0_spec_start = self.make_scale_input(self.frame_lengths[0], mask_start, mask_end, img.shape[-1]-1)
        scale1_spec_start = self.make_scale_input(self.frame_lengths[1], mask_start, mask_end, img.shape[-1]-1)
        scale2_spec_start = self.make_scale_input(self.frame_lengths[2], mask_start, mask_end, img.shape[-1]-1)
        scale3_spec_start = self.make_scale_input(self.frame_lengths[3], mask_start, mask_end, img.shape[-1]-1)
                
        gathered_img0 = self.make_gather_input(img, scale0_spec_start, self.frame_lengths[0])
        gathered_img1 = self.make_gather_input(img, scale1_spec_start, self.frame_lengths[1])
        gathered_img2 = self.make_gather_input(img, scale2_spec_start, self.frame_lengths[2])
        gathered_img3 = self.make_gather_input(img, scale3_spec_start, self.frame_lengths[3])

        scale0_output = self.scale0_discriminator(gathered_img0)
        scale1_output = self.scale1_discriminator(gathered_img1)
        scale2_output = self.scale2_discriminator(gathered_img2)
        scale3_output = self.scale3_discriminator(gathered_img3)
        multi_scale_output = torch.cat([scale0_output, scale1_output], -1)
        multi_scale_output = torch.cat([multi_scale_output, scale2_output], -1)
        multi_scale_output = torch.cat([multi_scale_output, scale3_output], -1)
        # return scale1_output, scale2_output, scale3_output, scale4_output
        return multi_scale_output