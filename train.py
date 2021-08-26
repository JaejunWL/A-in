from pathlib import Path
import argparse
import os
import warnings

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--save_folder', type = str, default = 'test', help = 'saving path that is a folder')
    parser.add_argument('--save_model', type = str, default = 'test', help = 'saving path that is a sub-folder')
    # parser.add_argument('--save_path', type = str, default = '/data2/personal/jaejun/inpainting/results/210723/1/models', help = 'saving path that is a folder')
    # parser.add_argument('--sample_path', type = str, default = '/data2/personal/jaejun/inpainting/results/210723/1/samples', help = 'training samples path that is a folder')
    parser.add_argument('--data_dir', type = str, default = '/data1/singing_inpainting/dataset', help = 'dataset directory')
    parser.add_argument('--add_datasets', type = str, default = [], help = 'datasets to use')
    # parser.add_argument('--add_datasets', type = str, default = ['artistscard-eng_singing', 'artistscard_singing', 'bighit_singing', 'changjo_singing', 'supertone_singing'], help = 'datasets to use')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "5, 6", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--load_name', type = str, default = '', help = '')
    parser.add_argument('--test', type=str, default=None, help='whether it is just test try or not')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 200, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 4e-4, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 5, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.8, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type = float, default = 1, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 10, help = 'the parameter of FML1Loss (perceptual loss)')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'the parameter of valid loss of AdaReconL1Loss; 0 is recommended')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--loss_region', type=int, default=3, help= '1 for whole loss, 2 for mask only loss, 3 for combination')
    parser.add_argument('--gp_weight', type=int, default=None, help='gradient penalty weight for wgan gp')
    parser.add_argument('--perceptual', type=str, default=None, help='whether use perceptual loss or not')
    parser.add_argument('--conformer_layer', type=int, default=4, help='use n th output layer of conformers')
    # Network parameters
    parser.add_argument('--stage_num', type = int, default = 1, help = 'two stage method or just only stage')
    parser.add_argument('--in_channels', type = int, default = 2, help = 'input real&complex spec + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 1, help = 'output real&complex spec')
    parser.add_argument('--latent_channels', type = int, default = 32, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    parser.add_argument('--discriminator', type = str, default = 'patch', help = 'the initialization gain')
    parser.add_argument('--discriminator_in_channel', type = int, default = 1, help = 'whether positional encoding is input to discriminator or not')
    parser.add_argument('--pos_enc', type=str, default=None, help = 'positinoal embedding')
    parser.add_argument('--msd_latent', type=int, default=64, help = 'latent channel dims of multi-scale discriminator (msd)')
    # Dataset parameters
    parser.add_argument('--mask_type', type = str, default = 'time', help = 'mask type')
    parser.add_argument('--mask_init', type = str, default = 'lerp', help = 'mask initialie point')
    parser.add_argument('--bbox', type = bool, default = False, help = 'whether use bbox shaping or not')
    parser.add_argument('--image_height', type = int, default = 1025, help = 'height of image')
    parser.add_argument('--image_width', type = int, default = 431, help = 'width of image')
    parser.add_argument('--input_length', type = int, default = 220500, help = 'input length (sample)')
    parser.add_argument('--spec_pow', type=int, default=2, help='1 for amplitude spec, 2 for power spec')
    parser.add_argument('--phase', type=int, default = 0, help = 'whether give phase information or not')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')

    # parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 120, help = 'margin of image for bbox mask')
    # parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    # parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    # parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
    opt = parser.parse_args()
    if opt.test != None:
        opt.save_folder='test'
    opt.save_path = os.path.join('/data2/personal/jaejun/inpainting/results', opt.save_folder, opt.save_model, 'models')
    opt.sample_path = os.path.join('/data2/personal/jaejun/inpainting/results', opt.save_folder, opt.save_model, 'samples')
    if opt.phase == 1:
        opt.in_channels = opt.in_channels + 1
        opt.out_channels = opt.out_channels + 1
    if opt.pos_enc != None:
        opt.in_channels = opt.in_channels + 1
    print(opt)
    
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Enter main function
    import trainer
    import wandb

    warnings.filterwarnings("ignore")

    wandb_save_dir = os.path.abspath(os.path.join(opt.save_path, '../'))

    wandb.init(project=opt.save_folder)
    # wandb.init(project="test")

    wandb.run.name = opt.save_model
    wandb.run.save(wandb_save_dir)
    wandb.config.update(opt)

    # if opt.gan_type == 'WGAN':
    trainer.WGAN_trainer(opt)
    # if opt.gan_type == 'LSGAN':
        # trainer.LSGAN_trainer(opt)
    
# 1 python train.py --save_folder=test --save_model=1 --gpu_ids='1,2' --gp_weight=10
# 3 python train.py --save_folder=test --save_model=2 --gpu_ids='3,4' --discriminator=jj --gan_type=GAN --lambda_gan=10
# 5 python train.py --save_folder=test --save_model=3 --gpu_ids='5,6' --discriminator=jj
# 7 python train.py --save_folder=test --save_model=4 --gpu_ids='7,8' --discriminator=jj --gp_weight=10
# 3 python train.py --save_folder=test --save_model=5 --gpu_ids='3,4' --gp_weight=10 --pos_enc=mel
# 5 python train.py --save_folder=210808 --save_model=7 --gpu_ids='7,8' --gp_weight=10 --pos_enc=cartesian
# 5 python train.py --save_folder=210808 --save_model=8 --gpu_ids='1,2' --pos_enc=mel --discriminator=jj --gan_type=GAN
# 5 python train.py --save_folder=210808 --save_model=9 --gpu_ids='9,10' --pos_enc=mel --discriminator=jj --gan_type=WGAN --gp_weight=10 --perceptual=1 --test=1

# python train.py --save_folder=210808 --save_model=10 --gpu_ids='9,10' --pos_enc=cartesian --gp_weight=10 --perceptual=1 --conformer_layer=4 --test=1 
# python train.py --save_folder=210808 --save_model=11 --gpu_ids='5,6' --pos_enc=cartesian --gp_weight=10 --perceptual=1 --conformer_layer=8 --test=1 
# 12 python train.py --save_folder=210808 --save_model=12 --gpu_ids='7,8' --pos_enc=cartesian --gp_weight=10 --perceptual=1 --conformer_layer=16 --test=1 
# 13 python train.py --save_folder=210808 --save_model=13 --gpu_ids='3,4' --pos_enc=cartesian --gp_weight=10 --perceptual=1 --conformer_layer=4 --loss_region=1 --test=1
# 114 python train.py --save_folder=210808 --save_model=114 --gpu_ids='5,6' --pos_enc=cartesian --gp_weight=10 --perceptual=1 --conformer_layer=4 --lambda_l1=1 --lambda_perceptual=100 --lambda_gan=10 --test=1
# 115 python train.py --save_folder=210808 --save_model=115 --gpu_ids='7,8' --pos_enc=cartesian --gp_weight=10 --perceptual=1 --conformer_layer=4 --lambda_l1=1 --lambda_perceptual=100 --lambda_gan=10 --test=1

# 1 python train.py --save_folder=210811 --save_model=1 --gpu_ids='1,2' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --test=1
# 2 python train.py --save_folder=210811 --save_model=2 --gpu_ids='3,4' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --test=1
# 3 python train.py --save_folder=210811 --save_model=3 --gpu_ids='5,6' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --discriminator_in_channel=2 --test=1
# 4 python train.py --save_folder=210811 --save_model=4 --gpu_ids='7,8' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --discriminator_in_channel=2 --test=1


# 4 python train.py --save_folder=210811 --save_model=5 --gpu_ids='5,6' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --discriminator_in_channel=2 --test=1
# 4 python train.py --save_folder=210811 --save_model=6 --gpu_ids='7,8' --pos_enc=cartesian --gp_weight=10 --discriminator=patch --msd_latent=32 --discriminator_in_channel=2 --test=1
# 4 python train.py --save_folder=210811 --save_model=7 --gpu_ids='1,2' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --mask_init=randn --test=1

# 4 python train.py --save_folder=210811 --save_model=8 --gpu_ids='5,6,7,8' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --mask_init=randn --batch_size=8 --latent_channels=48 --norm=bn --test=1


# python train.py --save_folder=210811 --save_model=7 --gpu_ids='3,4' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --mask_init=randn --bbox=True --test=1

# python train.py --save_folder=210820 --save_model=1 --gpu_ids='1,2' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --mask_init=randn --batch_size=8 --test=1


# python train.py --save_folder=210826 --save_model=1 --batch_size=8 --gpu_ids='1,2,3,4' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --mask_type=ctime --mask_init=randn --test=1
# python train.py --save_folder=210826 --save_model=2 --batch_size=8 --gpu_ids='5,6,7,8' --pos_enc=cartesian --gp_weight=10 --discriminator=multi --mask_type=cbtime --mask_init=randn --test=1

