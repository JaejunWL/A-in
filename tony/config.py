import argparse
import os
import random
import sys
import torch
import yaml
import numpy as np
import wandb

parser = argparse.ArgumentParser()

def get_run_script():
    run_script = 'python'
    for e in sys.argv:
        run_script += (' ' + e)
    return run_script


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_params(params):
    if params.tags is None or params.num_epochs < 10:
        assert False

def get_args():
    params = parser.parse_args()
    params.run_script = get_run_script()

    # tag&save
    params.tags = [e for e in params.tags.split(',')] if params.tags is not None else ['test']
    params.tags.append(params.name)

    # random_seed
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed_all(params.random_seed)
    random.seed(params.random_seed)

    params.weight_path = f'{params.output_dir}{params.name}/ckpt/'
    params.temp_weight_path = f'{params.output_dir}{params.name}/ckpt_temp/'
    os.makedirs(params.weight_path, exist_ok=True)
    os.makedirs(params.temp_weight_path, exist_ok=True)
    os.makedirs(f'{params.output_dir}{params.name}/log/', exist_ok=True)

    return params

def print_args(params, save=False):
    info = '\n[args]\n'
    for sub_args in parser._action_groups:
        if sub_args.title in ['positional arguments', 'optional arguments']:
            continue
        size_sub = len(sub_args._group_actions)
        info += f'  {sub_args.title} ({size_sub})\n'
        for i, arg in enumerate(sub_args._group_actions):
            prefix = '-'
            info += f'      {prefix} {arg.dest:20s}: {getattr(params, arg.dest)}\n'
    info += '\n'
    print(info)

    if save:
        record_path = f"{args.output_dir}{args.name}/args.txt"
        f = open(record_path, 'w')
        np.savetxt(f, [info], delimiter=" ", fmt="%s")
        f.close()




""" Data Parameters"""

''' Dataset names
    1. "marg"
    2. "artistscard"
    -1. "ALL"   =>   using all available datasets
'''
dataset_names = {1:"marg", 2:"artistscard", -1:"all"}

''' dataprep version and its number of phoneme labels
    1. korean : 46
    2. universal : 51
'''
num_phoneme_labels = {'korean':46, 'universal': 51}


""" set your configurations """

# universal_conformer_large_cont
# using_gpu = "13, 14"
# master_port = "8888"

# conformer_medium_mel_only
# using_gpu = "11, 12"
# master_port = "8889"

# conformer_small_mel_only
using_gpu = "9, 10"
master_port = "8890"

# conformer_medium_re - inference
# using_gpu = "2"
# master_port = "8891"

gpu_size = len(using_gpu.split(","))
batch_size = gpu_size * 32
batch_size_test = gpu_size * 1
worker_size = gpu_size * 4

using_dataset_idx = -1
dataprep_version = 'korean'      # 1. korean 2. universal
est_target = 'both'                 # 1. both   2. pitch   3. text


# TODO : 
# !!!normalizing log mel scale!!!

# evaluations
# 1. full song inference
# 2. to MIDI -> onset, offset, pitch evaluation


''' detailed configurations '''
reload_args = parser.add_argument_group('Reloading args')
reload_args.add_argument('--reload', type=str2bool, default=False)
reload_args.add_argument('--inference_only', type=str2bool, default=False)
reload_args.add_argument('--manual_reload', type=str2bool, default=False)
reload_args.add_argument('--manual_reload_name', type=str, default="universal_conformer_large", help="reloading from different path")
reload_args.add_argument('--reload_model', type=str, default=["MLP"], help="reloading network architecture")
reload_args.add_argument('--reload_epoch', type=str, default=None, help="reloading check point epoch number. default=None means loading latest epoch")

base_args = parser.add_argument_group('Base args')
base_args.add_argument('--name', type=str, default='conformer_small_mel_only', help="experiment name")
base_args.add_argument('--data_dir', type=str, default="/data/Healthcare/ASM/dataset/singing/")
base_args.add_argument('--output_dir', type=str, default="/workspace/Korean_AST/results/")

wandb_args = parser.add_argument_group('wandb args')
wandb_args.add_argument('--use_wandb', type=str2bool, default='True')
wandb_args.add_argument('--project', type=str, default="Korean_Automatic_Singing_Transription")
wandb_args.add_argument('--tags')

train_args = parser.add_argument_group('Train args')
train_args.add_argument('--est_target', type=str, default=est_target, help="targets to estimate. possible methods are : 1. both   2. pitch   3. text")
train_args.add_argument('--random_seed', type=int, default=2)
train_args.add_argument('--num_epochs', type=int, default=300)
train_args.add_argument('--train_batch', type=int, default=batch_size)
train_args.add_argument('--test_batch', type=int, default=batch_size_test)
train_args.add_argument('--valid_per_epoch', type=int, default=1)
train_args.add_argument('--test_per_epoch', type=int, default=1)
train_args.add_argument('--using_loss_pitch', type=str, default=["softdtw", ], help="check file 'loss.py' for available loss functions")
train_args.add_argument('--using_loss_text', type=str, default=["softdtw", ], help="check file 'loss.py' for available loss functions")
train_args.add_argument('--using_metrics', type=str, default=["accuracy", "f1_weighted", "softdtw"], help="check file 'loss.py' for available evaluation metrics")

data_args = parser.add_argument_group('Data args')
data_args.add_argument('--using_dataset', type=str, default=dataset_names[using_dataset_idx], help="using dataset")
data_args.add_argument('--dataprep_version', type=str, default=dataprep_version)
data_args.add_argument('--using_features', type=str, default=['mel',], help="using feature inputs for the network input")
data_args.add_argument('--num_pitch_label', type=int, default=88)
data_args.add_argument('--num_phoneme_label', type=int, default=num_phoneme_labels[dataprep_version])

pretrained_args = parser.add_argument_group('Pretrained args')
pretrained_args.add_argument('--pretrained_model_weights_dir', type=str, default='/data4/Hyundai_Stress_Recognition/pretrained_model_weights/')
pretrained_args.add_argument('--using_pretrained_model', type=str, default='wav2vec')
pretrained_args.add_argument('--extraction_method', type=str, default="none", help="possible extraction methods are : 1. max   2. mean   3. none")
pretrained_args.add_argument('--pretrained_model_version_aud', type=str, default="xlsr", help="possible sub pre-trained model for Wav2Vec 2.0 : 1. xlsr   2. ")
pretrained_args.add_argument('--wav2vec_feature_aud', type=str, default="c", help="possible feature extracted from model for Wav2Vec 2.0 : 1. z   2. c")
pretrained_args.add_argument('--transformer_layer', type=int, default=14, help="using output representation of the desired transformer layer")

acoustic_args = parser.add_argument_group('Acoustic args')
acoustic_args.add_argument('--front_end_using_spec', type=str, default=["log_mel"], help="using input features; possible features are: 1. cplx   2. mag_phase   3. mag   4. mel")
acoustic_args.add_argument('--channel', type=str, default="mono", help="using audio channels: 1. mono   2. stereo   ")
acoustic_args.add_argument('--window', type=str, default="hann", help="possible window types: 1. hann   2. hamming   ")
acoustic_args.add_argument('--n_fft', type=int, default=1024)
acoustic_args.add_argument('--hop_length', type=int, default=None)
acoustic_args.add_argument('--win_length', type=int, default=None)
acoustic_args.add_argument('--sample_rate', type=int, default=22050)
acoustic_args.add_argument('--segment_length', type=int, default=1024*60)
acoustic_args.add_argument('--mel_channel_num', type=int, default=128)
acoustic_args.add_argument('--mel_time_factor', type=int, default=4)

network_args = parser.add_argument_group('Network args')
network_args.add_argument('--network_arc', type=str, default='Conformer', help="possible network architectures are: 1. Conformer   2. ")
network_args.add_argument('--network_options', type=str, default='small', help="network options for each architecture: see configs.yaml for detailed settings")
network_args.add_argument('--dropout_rate', type=float, default=0.0)

hyperparam_args = parser.add_argument_group('Hyperparameter args')
hyperparam_args.add_argument('--optimizer', type=str, default='Adam', help="possible optimizers are: 1. Adam   2. RAdam")
hyperparam_args.add_argument('--learning_rate', type=float, default=1e-4)
hyperparam_args.add_argument('--decay_factor', default=0.1, type=float, help="leraning rate decay factor")
hyperparam_args.add_argument('--weight_decay', default=0.0, type=float, help="leraning rate decay factor")
hyperparam_args.add_argument('--patience', default=2, type=int, help='patience')
hyperparam_args.add_argument('--eps', default=1e-7, type=int, help="epsilon value for preventing 'nan' values")

gpu_args = parser.add_argument_group('GPU args')
gpu_args.add_argument('--master_address', type=str, default='127.0.0.1', help="using master address number")
gpu_args.add_argument('--using_gpus', type=str, default=using_gpu, help="names of using GPU")
gpu_args.add_argument('--master_port', type=str, default=master_port, help="using port number")
gpu_args.add_argument('--n_nodes', default=1, type=int)
gpu_args.add_argument('--workers', default=worker_size, type=int)
gpu_args.add_argument('--rank', default=0, type=int, help='ranking within the nodes')


args = get_args()

''' Additional Handling of Configurations '''
# load network configurations
with open('configs.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
args.cfg = configs[args.network_arc][args.network_options]

# wav2vec feature dimension
if args.wav2vec_feature_aud.lower()=='c':
    args.wav2vec_feat_dim = 1024
elif args.wav2vec_feature_aud.lower()=='z':
    args.wav2vec_feat_dim = 512

# acoustic configurations
args.hop_length = args.hop_length if args.hop_length else args.n_fft//4
args.win_length = args.win_length if args.win_length else args.n_fft

# adding a label for CTC loss computation
if 'ctc' in args.using_loss_text:
    args.num_phoneme_label += 1

# handling loss function for single tasked model
if args.est_target.lower()!='both':
    if args.est_target.lower()=='pitch':
        args.using_loss_text = []
    elif args.est_target.lower()=='text':
        args.using_loss_pitch = []


if __name__ == '__main__':
    args = get_args()
    print_args(args)