""" 
    Implementation of transcribers for the task
    'Korean Automatic Singing Transcription'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from conformer import ConformerBlock

from tony.tony_network.tony_utils import *



# Pitch-based Singing Transcriber
class Pitch_Transcriber(nn.Module):
    def __init__(self, config):
        pass


    # network forward operation
    def forward(self, src_audio):

        return




# Phoneme-based Singing Transcriber
class Phoneme_Transcriber(nn.Module):
    def __init__(self, config):

        input_dim = 80
        num_classes = 15
        network_arc = "conformer"
        
        if network_arc=="conformer":
            self.model = Conformer(num_classes=num_classes, input_dim=input_dim, \
                                    encoder_dim=config["encoder_dim"], decoder_dim=["decoder_dim"], \
                                    num_encoder_layers=config["num_encoder_layers"], num_decoder_layers=config["num_decoder_layers"], \
                                    num_attention_heads=config["num_attention_heads"], \
                                    conv_kernel_size=config["conv_kernel_size"])


    # network forward operation
    def forward(self, src_audio):

        return



# Singing Transcriber which estimates both the pitch and phoneme sequence from the input audio
class Singing_Transcriber(nn.Module):
    def __init__(self, config):
        pass


    # network forward operation
    def forward(self, src_audio):

        return


# Conformer
class Multi_Input_Conformer(nn.Module):
    def __init__(self, args, drop_rate=0.1):
        super(Multi_Input_Conformer, self).__init__()
        config = args.cfg
        feature_dimension_dict = {'mel':args.mel_channel_num, 'wav2vec':args.wav2vec_feat_dim}
        self.using_features = args.using_features
        assert self.using_features, "at least a single using feature for the network input must be given"

        ''' Convolution Subsampling Module '''
        self.conv_subsample_modules = nn.ModuleDict()
        subsample_combined_out_dimension = 0
        for cur_feature in self.using_features:
            cur_out_dim = config["subsample_out_dim"]
            if cur_feature=="wav2vec":
                cur_out_dim //= 16
            cur_subsample_sequential = nn.Sequential(
                nn.Conv2d(1, cur_out_dim, kernel_size=config["subsample_kernel_size"], stride=config["subsample_stride_size"], padding=config["subsample_padding_size"]),
                nn.ReLU(),
                nn.Conv2d(cur_out_dim, cur_out_dim, kernel_size=config["subsample_kernel_size"], stride=config["subsample_stride_size"], padding=config["subsample_padding_size"]),
                nn.ReLU(),
            )
            self.conv_subsample_modules[cur_feature] = cur_subsample_sequential
            subsample_combined_out_dimension += (cur_out_dim * feature_dimension_dict[cur_feature] // (config["subsample_stride_size"]*2))
        
        # project subsampled outputs to match input of the conformer block
        self.input_projection = nn.Sequential(
			Linear(subsample_combined_out_dimension, config["encoder_dim"]),
			nn.Dropout(p=drop_rate),
		)

        ''' Conformer Encoder and Decoder '''
        # encoder
        self.conformer_blocks = nn.ModuleList()
        for i in range(config["nums_encoder_layers"]):
            self.conformer_blocks.append(ConformerBlock(dim=config["encoder_dim"], dim_head=config["encoder_dim"]//config["num_attention_heads"], \
                                                        heads=config["num_attention_heads"], ff_mult=config["feed_forward_expansion_rate"], \
                                                        conv_expansion_factor=config["conv_expansion_factor"], conv_kernel_size=config["conv_kernel_size"], \
                                                        attn_dropout=config["attention_dropout_rate"], ))
        
        # decoder
        self.decoder = nn.LSTM(config["encoder_dim"], config["decoder_dim"], config["num_decoder_layers"], batch_first=True) \
                                                                                        if config["num_decoder_layers"] > 0 else None

        # final decision layer
        final_projection_input_dim = config["decoder_dim"] if self.decoder else config["encoder_dim"]
        self.est_target = args.est_target
        if self.est_target=="both":
            self.num_pitch_label = args.num_pitch_label
            final_output_dim = args.num_pitch_label + args.num_phoneme_label
        elif self.est_target=="pitch":
            final_output_dim = args.num_pitch_label
        elif self.est_target=="text":
            final_output_dim = args.num_phoneme_label
        self.final_projection = Linear(final_projection_input_dim, final_output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        

    def forward(self, input_features):

        ''' Convolutional Subsampling '''
        subsample_outputs_combined = []
        for input_index, cur_key in enumerate(self.using_features):
            assert cur_key in input_features, f"No feature named '{cur_key}' in input\n   check the configuration and solver file"
            # subsample module output shape : batch x channels x dimension x time
            cur_subsample_output = self.conv_subsample_modules[cur_key](input_features[cur_key].unsqueeze(1))
            batch_size, channels, dim, lengths = cur_subsample_output.size()
            # reshaped to : batch x dimension x time
            cur_subsample_output = cur_subsample_output.reshape(batch_size, channels*dim, lengths)
            # combine results
            subsample_outputs_combined = cur_subsample_output if input_index==0 \
                                    else torch.cat((subsample_outputs_combined, cur_subsample_output), axis=1)

        outputs = self.input_projection(subsample_outputs_combined.permute(0,2,1))

        ''' Encoding '''
        for cur_enc_layer in self.conformer_blocks:
            outputs = cur_enc_layer(outputs)
        
        ''' Decoding '''
        if self.decoder:
            outputs, _ = self.decoder(outputs)

        # decision layer
        outputs = self.final_projection(outputs)
        if self.est_target=='both':
            pitch_est = self.softmax(outputs[:,:,:self.num_pitch_label])
            text_est = self.softmax(outputs[:,:,self.num_pitch_label:])
            text_est_lsm = self.log_softmax(outputs[:,:,self.num_pitch_label:])
            return {'pitch':pitch_est, 'text':text_est, 'text_lsm':text_est_lsm}
        elif self.est_target=='text':
            text_est = self.softmax(outputs)
            text_est_lsm = self.log_softmax(outputs)
            return {'text':text_est, 'text_lsm':text_est_lsm}
        elif self.est_target=='pitch':
            pitch_est = self.softmax(outputs)
            return {'pitch': pitch_est}


# Conformer
class jj_Conformer(nn.Module):
    def __init__(self, args, drop_rate=0.1):
        super(jj_Conformer, self).__init__()
        config = args.cfg
        feature_dimension_dict = {'mel':args.mel_channel_num}
        self.using_features = args.using_features
        assert self.using_features, "at least a single using feature for the network input must be given"

        ''' Convolution Subsampling Module '''
        self.conv_subsample_modules = nn.ModuleDict()
        subsample_combined_out_dimension = 0
        for cur_feature in self.using_features:
            cur_out_dim = config["subsample_out_dim"]
            cur_subsample_sequential = nn.Sequential(
                nn.Conv2d(1, cur_out_dim, kernel_size=config["subsample_kernel_size"], stride=config["subsample_stride_size"], padding=config["subsample_padding_size"]),
                nn.ReLU(),
                nn.Conv2d(cur_out_dim, cur_out_dim, kernel_size=config["subsample_kernel_size"], stride=config["subsample_stride_size"], padding=config["subsample_padding_size"]),
                nn.ReLU(),
            )
            self.conv_subsample_modules[cur_feature] = cur_subsample_sequential
            subsample_combined_out_dimension += (cur_out_dim * feature_dimension_dict[cur_feature] // (config["subsample_stride_size"]*2))
        
        # project subsampled outputs to match input of the conformer block
        self.input_projection = nn.Sequential(
			Linear(subsample_combined_out_dimension, config["encoder_dim"]),
			nn.Dropout(p=drop_rate),
		)

        ''' Conformer Encoder and Decoder '''
        # encoder
        self.conformer_blocks = nn.ModuleList()
        for i in range(config["nums_encoder_layers"]):
            self.conformer_blocks.append(ConformerBlock(dim=config["encoder_dim"], dim_head=config["encoder_dim"]//config["num_attention_heads"], \
                                                        heads=config["num_attention_heads"], ff_mult=config["feed_forward_expansion_rate"], \
                                                        conv_expansion_factor=config["conv_expansion_factor"], conv_kernel_size=config["conv_kernel_size"], \
                                                        attn_dropout=config["attention_dropout_rate"], ))

    def forward(self, input_features):
        ''' Convolutional Subsampling '''
        subsample_outputs_combined = []
        for input_index, cur_key in enumerate(self.using_features):
            # print('#2', len(cur_key), cur_key.shape)

            assert cur_key in input_features, f"No feature named '{cur_key}' in input\n   check the configuration and solver file"
            # subsample module output shape : batch x channels x dimension x time
            cur_subsample_output = self.conv_subsample_modules[cur_key](input_features[cur_key].unsqueeze(1))
            batch_size, channels, dim, lengths = cur_subsample_output.size()
            # reshaped to : batch x dimension x time
            cur_subsample_output = cur_subsample_output.reshape(batch_size, channels*dim, lengths)
            # combine results
            subsample_outputs_combined = cur_subsample_output if input_index==0 \
                                    else torch.cat((subsample_outputs_combined, cur_subsample_output), axis=1)
        
        # print(subsample_outputs_combined.shape)
        outputs = self.input_projection(subsample_outputs_combined.permute(0,2,1))
        ''' Encoding '''
        for cur_enc_layer in self.conformer_blocks:
            outputs = cur_enc_layer(outputs)
        return outputs


if __name__ == '__main__':
    ''' check model I/O shape '''
    from config import args
    import yaml
    with open('configs.yaml', 'r') as f:
        configs = yaml.load(f)


    batch_size = 64
    time_length = args.segment_length // args.hop_length
    input = {'wav2vec':torch.rand(batch_size, args.wav2vec_feat_dim, time_length), 'mel':torch.rand(batch_size, args.mel_channel_num, time_length)}

    model_arc = "Conformer"
    model_options = "default"
    config = configs[model_arc][model_options]
    print(config)
    
    # network = Phoneme_Transcriber(config)
    # pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters : {pytorch_total_params}")

    # output = network(input, ref_embedding)
    # print(f"Output Shape : {output.shape}\n")

    from conformer_transcriber import *
    network = Transcriptor()
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {pytorch_total_params}")

    network = Multi_Input_Conformer(args)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {pytorch_total_params}")

    p_out, t_out = network(input)
    print(f"Output Shape : pitch={p_out.shape}   text={t_out.shape}")

