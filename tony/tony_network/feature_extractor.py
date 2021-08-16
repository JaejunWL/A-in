"""
    Implementation of feature extractors of audio pre-trained models

"""
import os
import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence



# function to return feature extractors
def retrieve_feature_extractor(args):
    if args.using_pretrained_model.lower()=="wav2vec":
        feature_extractor_aud = Wav2Vec2FeatureExtractor(args=args, checkpoint_dir=args.pretrained_model_weights_dir, \
                                                            model_version=args.pretrained_model_version_aud, \
                                                            extraction_method=args.extraction_method, \
                                                            gpu_num=args.gpu)
                                                            
    return feature_extractor_aud


''' Wav2Vec 2.0 '''
class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self, args, checkpoint_dir, model_version, using_feature='z', extraction_method="None", gpu_num=0):
        super().__init__()
        ### install requirements for Wav2Vec 2.0 feature extractor ###
        # $ git clone https://github.com/pytorch/fairseq
        # $ cd fairseq
        # $ pip install --editable ./
        import fairseq

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_num}")
        self.device = device

        self.using_feature = args.wav2vec_feature_aud
        self.extraction_method = args.extraction_method

        # set checkpoint dir
        if model_version.lower()=='xlsr':
            model_pt_name = "xlsr_53_56k.pt"
        ckpt_path = checkpoint_dir + model_pt_name

        # set using feature : rather to use 'z' or 'c' feature
        # reload model
        wav2vec_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        wav2vec_model = wav2vec_model[0]

        # send to GPU
        self.model = wav2vec_model.to(self.device)

        ori_freq = args.sample_rate
        new_freq = 16000

        # resampler
        self.resample_func = torchaudio.transforms.Resample(orig_freq=ori_freq, new_freq=new_freq) if ori_freq!=new_freq else None
        self.resampling_ratio = new_freq / ori_freq


    def forward(self, audio_data, audio_length, skip_resample=False):
        # resample if required and set expected output length 
        audio_length = torch.FloatTensor(audio_length)
        if not skip_resample and self.resample_func:
            audio_data = self.resample_func(audio_data)
            audio_length *= self.resampling_ratio
        # receptive field = 400 samples   &   stride = 320 samples
        audio_length = (audio_length.type(torch.IntTensor) - 80) // 320

        # inference
        self.model.eval()
        with torch.no_grad():
            if self.using_feature=='z':
                feature = self.model.feature_extractor(audio_data)
                time_dim = 2
                feat_dim = 1
            elif self.using_feature=='c':
                feature = self.model(audio_data, features_only=True, mask=False)['x']
                time_dim = 1
                feat_dim = 2

        # returning method
        if self.extraction_method=="max":
            mask = torch.arange(feature.shape[time_dim])[None, :] < audio_length[:, None]
            mask = (mask*-99).unsqueeze(feat_dim).to(self.device)
            fin_feature = torch.max(feature+mask, time_dim).values

        elif self.extraction_method=="mean":
            mask = torch.arange(feature.shape[time_dim])[None, :] < audio_length[:, None]
            mask = mask.unsqueeze(feat_dim).to(self.device)
            masked_feature = feature*mask
            fin_feature = masked_feature.sum(dim=time_dim) / mask.sum(dim=time_dim)

        elif self.extraction_method=="pad":
            fin_feature = pack_padded_sequence(feature, audio_length, batch_first=True, enforce_sorted=False)

        elif self.extraction_method=="none":
            fin_feature = feature

        return fin_feature





# check feature extractor
if __name__ == '__main__':
    """
    Test code for Wav2Vec feature extractor
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '12'

    currentdir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(currentdir+"/../../model/")
    from config import args

    # input setup
    batch_size = 32
    max_len = 16000*3
    # input_length = list(torch.randint(100, max_len, (batch_size,)))
    input_length = [max_len for i in range(batch_size)]
    
    # pad audio
    audio_list = []
    for cur_input_len in input_length:
        audio_list.append(torch.rand(cur_input_len))
        # audio_list.append(torch.zeros(torch.randint(1,max_len,(1,))))
    from torch.nn.utils.rnn import pad_sequence
    padded_aud = pad_sequence(audio_list, batch_first=True)
    
    # ckpt_dir = "/data1/Hyundai_Stress_Recognition/pretrained_model_weights/"
    ckpt_dir = "/data/Hyundai_Stress_Recognition/pretrained_model_weights/"

    ''' Wav2Vec '''
    print('checking Wav2Vec feature extractor..')
    # xlsr
    wav2vec_fe = Wav2Vec2FeatureExtractor(args=args, checkpoint_dir=ckpt_dir, using_feature='c', model_version='xlsr', extraction_method='max')
    output = wav2vec_fe.forward(padded_aud.cuda(), input_length)
    print(output.shape)

