import easydict
import yaml

def make_tony_args(opt):
    config_dir = 'tony/configs.yaml'
    with open(config_dir, 'r') as f:
        configs = yaml.load(f)

    tony_args = easydict.EasyDict({
        "mel_channel_num": 128,
        "using_features": ['mel'],
        "est_target": 'both',
        "num_pitch_label": 88,
        "num_phoneme_label": 46,
        "model_arc": "Conformer",
        "model_options": "small",
        "segment_length": 1024*60,
        "n_fft" : 1024,
        "win_length" : 1024,
        "hop_length" : 256,
        "eps": 1e-7,
        "channel": 'mono',
        "window" : 'hann',
        "sample_rate": 22050,
        "front_end_using_spec": ["log_mel"],
        })

    model_arc = tony_args.model_arc
    model_options = tony_args.model_options
    config = configs[model_arc][model_options]
    config['nums_encoder_layers'] = opt.conformer_layer

    tony_args.cfg = config
    
    return tony_args