import torch
import torch.nn as nn
from conformer import ConformerBlock
import torch.nn.init as init


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)


class Transcriptor(nn.Module):
	def __init__(self, n_blocks=16, w2v_dim=768, mel_dim=80, out_dim=128, drop_rate=0.1):
		super(Transcriptor, self).__init__()
		self.n_blocks = n_blocks
		self.sequential_t = nn.Sequential(
			nn.Conv2d(1, out_dim//16, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(out_dim//16, out_dim//16, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
		)
		self.sequential_m = nn.Sequential(
			nn.Conv2d(1, out_dim, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
		)
		self.input_projection = nn.Sequential(
			Linear(4608, 256),
			nn.Dropout(p=drop_rate),
		)
		self.conformer_blocks = nn.ModuleList()
		for i in range(self.n_blocks):
			self.conformer_blocks.append(ConformerBlock(dim=256, dim_head=64, heads=4, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=31,attn_dropout=0.1, ))
		# self.final_projection = Linear(256, 88+60)
		self.final_projection = Linear(256, 88+46)
		self.softmax = torch.nn.Softmax(dim=-1)
	def forward(self, feats):
		t_in = feats['ling']
		t_enc = self.sequential_t(t_in.unsqueeze(1))
		batch_size, channels, dim, lengths = t_enc.size()
		t_enc = t_enc.permute(0,3,1,2)
		t_enc = t_enc.contiguous().view(batch_size, lengths, channels*dim)
		t_enc = t_enc.permute(0,2,1) # b, dim, time
		m_in = feats['mel']
		m_enc = self.sequential_m(m_in.unsqueeze(1))
		batch_size, channels, dim, lengths = m_enc.size()
		m_enc = m_enc.permute(0,3,1,2)
		m_enc = m_enc.contiguous().view(batch_size, lengths, channels*dim)
		m_enc = m_enc.permute(0,2,1) # b, dim, time
		outputs = self.input_projection(torch.cat((t_enc, m_enc), axis=1).permute(0,2,1))
		for layer in self.conformer_blocks:
			outputs = layer(outputs)
		outputs = self.final_projection(outputs)
		p_est = self.softmax(outputs[:,:,:88])
		t_est = self.softmax(outputs[:,:,88:])
		return p_est, t_est



# model_parameters = filter(lambda p: p.requires_grad, a.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])