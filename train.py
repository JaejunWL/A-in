from pathlib import Path
import argparse
import os

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--save_path', type = str, default = '/data2/personal/jaejun/inpainting/results/210723/1/models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = '/data2/personal/jaejun/inpainting/results/210723/1/samples', help = 'training samples path that is a folder')
    parser.add_argument('--data_dir', type = str, default = '/data1/singing_inpainting/dataset', help = 'dataset directory')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "7, 8", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    # parser.add_argument('--load_name', type = str, default = '/data2/personal/jaejun/inpainting/results/210717/time/models/deepfillv2_WGAN_epoch30_batchsize4.pth', help = '')
    parser.add_argument('--load_name', type = str, default = '', help = '')

    # Training parameters
    parser.add_argument('--epochs', type = int, default = 200, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 4e-4, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 5, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type = float, default = 100, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 10, help = 'the parameter of FML1Loss (perceptual loss)')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'the parameter of valid loss of AdaReconL1Loss; 0 is recommended')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--loss_mask', type=str, default=None, help= 'True if loss normalization by sum(mask region)')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 2, help = 'input real&complex spec + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 1, help = 'output real&complex spec')
    parser.add_argument('--latent_channels', type = int, default = 32, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256", help = 'the training folder')
    parser.add_argument('--mask_type', type = str, default = 'time', help = 'mask type')
    parser.add_argument('--mask_init', type = str, default = 'lerp', help = 'mask initialie point')
    parser.add_argument('--image_height', type = int, default = 1024, help = 'height of image')
    parser.add_argument('--image_width', type = int, default = 428, help = 'width of image')
    parser.add_argument('--input_length', type = int, default = 220500, help = 'input length (sample)')

    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    # parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 120, help = 'margin of image for bbox mask')
    # parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    # parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    # parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
    opt = parser.parse_args()
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
    # wandb.init(project="210723")
    wandb.init(project="test")

    wandb.run.name = '1'
    wandb.run.save(Path(opt.save_path).parent)
    wandb.config.update(opt)

    if opt.gan_type == 'WGAN':
        trainer.WGAN_trainer(opt)
    if opt.gan_type == 'LSGAN':
        trainer.LSGAN_trainer(opt)
    