import functools
from dataclasses import dataclass

from torch import nn
from torchtyping import TensorType

from src.Networks.CycleGAN.networks import define_G, ResnetBlock, get_norm_layer, _get_activation
from src.Networks.EncoderDecoder import BaseEncoderDecoder, get_activation
from src.Networks.MLP_EncoderDecoder import BaseEncoderDecoder as BaseEncoderDecoder_MLP
from src.Networks.ReshapeLayer import ReshapeLayer


def full_cnn_network(input_dimension, layers, latent_space_function):
	input_Encoder_Decoder = BaseEncoderDecoder(
		in_channels=input_dimension,
		layers=layers,
		out_dim=input_dimension,
		latent_space_function=latent_space_function,
	)
	return input_Encoder_Decoder


def _get_linking_part(_linking_part, cnn_link, bottleneck_size, latent_space_function):
	link = []
	if not cnn_link:
		link.append(nn.Flatten())
	for in_c, out_c in zip(_linking_part, _linking_part[1:]):
		if cnn_link:
			link.append(nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
			link.append(nn.BatchNorm2d(in_c))
		else:
			link.append(nn.Linear(in_c * 2 * 2, out_c * 2 * 2, bias=True))
			link.append(nn.BatchNorm1d(out_c * 2 * 2))
			link.append(ReshapeLayer([-1, out_c, bottleneck_size, bottleneck_size]))

		link.append(nn.LeakyReLU())
	link[-1] = get_activation(latent_space_function)
	return nn.Sequential(*link)


def _get_mlp_linking_part(linking_part, latent_space_function):
	linking_part_list = []
	for in_c, out_c in zip(linking_part, linking_part[1:]):
		linking_part_list.append(nn.Flatten())
		linking_part_list.append(nn.Linear(in_c, out_c, bias=True))
		linking_part_list.append(nn.BatchNorm1d(out_c))
		linking_part_list.append(nn.LeakyReLU())
	linking_part_list[-1] = get_activation(latent_space_function)
	return nn.Sequential(*linking_part_list)


def _get_mlp_network(output_dimension, layers_MLP, latent_space_function):
	return BaseEncoderDecoder_MLP(
		in_dim=output_dimension,
		layers=layers_MLP,
		out_dim=output_dimension,
		latent_space_function=latent_space_function,
	)


def get_full_cnn_native(params: dict):
	in_part  = full_cnn_network(params['in_dim'], params['layers'], params['latent_space_function'])
	in_part_decoder = nn.Sequential(in_part.decoder, _get_activation(params['o_activation_x']))

	out_part = full_cnn_network(params['out_dim'], params['layers'], params['latent_space_function'])
	out_part_decoder = nn.Sequential(out_part.decoder, _get_activation(params['o_activation_y']))

	nb_conv = len(params['layers'])
	down_scale_factor = 2 ** nb_conv
	bottleneck_resolution = params['image_res'] // down_scale_factor

	middle1 = _get_linking_part(params['link']      , params['cnn_link'], bottleneck_resolution, params['latent_space_function'])
	middle2 = _get_linking_part(params['link'][::-1], params['cnn_link'], bottleneck_resolution, params['latent_space_function'])

	x_to_x = nn.Sequential()
	x_to_x.add_module('encoder', in_part.encoder)
	x_to_x.add_module('decoder', in_part_decoder)

	y_to_y = nn.Sequential()
	y_to_y.add_module('encoder', out_part.encoder)
	y_to_y.add_module('decoder', out_part_decoder)

	link_x_to_y = middle1
	link_y_to_x = middle2

	return [x_to_x, y_to_y, link_x_to_y, link_y_to_x]


def get_cnn_mlp_native(params: dict):
	in_part = full_cnn_network(params['in_dim'], params['layers_cnn'], params['latent_space_function'])
	in_part_decoder = nn.Sequential(in_part.decoder, _get_activation(params['o_activation_x']))

	out_part = _get_mlp_network(params['out_dim'], params['layers_mlp'], params['latent_space_function'])
	out_part_decoder = nn.Sequential(out_part.decoder, _get_activation(params['o_activation_y']))

	nb_conv = len(params['layers_cnn'])
	down_scale_factor = 2 ** nb_conv
	bottleneck_resolution = params['image_res'] // down_scale_factor

	middle1 = _get_mlp_linking_part(params['link']      , params['latent_space_function'])
	middle2 = _get_mlp_linking_part(params['link'][::-1], params['latent_space_function'])

	x_to_x = nn.Sequential()
	x_to_x.add_module('encoder', in_part.encoder)
	x_to_x.add_module('decoder', in_part_decoder)

	y_to_y = nn.Sequential()
	y_to_y.add_module('encoder', out_part.encoder)
	y_to_y.add_module('decoder', out_part_decoder)

	link_x_to_y = middle1
	link_y_to_x = nn.Sequential(middle2, ReshapeLayer([-1, bottleneck_resolution, bottleneck_resolution], keep_bs=True))

	return [x_to_x, y_to_y, link_x_to_y, link_y_to_x]


#####

def getResnetBlock(dim: int, padding_type: str, norm_layer: str, use_dropout: bool) -> nn.Module:
	norm_layer = get_norm_layer(norm_type=norm_layer)
	if type(norm_layer) == functools.partial:
		use_bias = norm_layer.func == nn.InstanceNorm2d
	else:
		use_bias = norm_layer == nn.InstanceNorm2d
	block = ResnetBlock(dim=dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
	return block


def get_cnn_link(params):
	link = getResnetBlock(dim=params['latent_resolution'], padding_type='reflect', norm_layer=params['norm_layer'], use_dropout=params['dropout'])
	return link


def get_full_cnn_resnet(params: dict):
	x_to_x = define_G(input_nc=params['in_dim'], output_nc=params['in_dim'], ngf=params['ngf'], netG=params['netG'],
	                  norm=params['norm_layer'], use_dropout=params['dropout'], init_type=params['init_type'],
	                  init_gain=params['init_gain'], o_activation=params['o_activation_x'])

	y_to_y = define_G(input_nc=params['out_dim'], output_nc=params['out_dim'], ngf=params['ngf'], netG=params['netG'],
	                  norm=params['norm_layer'], use_dropout=params['dropout'], init_type=params['init_type'],
	                  init_gain=params['init_gain'], o_activation=params['o_activation_y'])

	link_x_to_y = get_cnn_link(params)
	link_y_to_x = get_cnn_link(params)

	return [x_to_x, y_to_y, link_x_to_y, link_y_to_x]


class cnn_2_mlp_link(nn.Module):
	def __init__(self, params):
		"""
		input shape is 256x16x16
		output shape is 68*2
		"""
		super(cnn_2_mlp_link, self).__init__()
		norm_layer = get_norm_layer(norm_type=params['norm_layer'])
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.model_cnn = nn.Sequential(
			# input shape is 256x16x16
			nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=1, bias=use_bias),
			# output shape is [batch, 256, 7, 7]
			norm_layer(256),
			nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=1, bias=use_bias),
			# output shape is [batch, 256, 3, 3]
			norm_layer(256),
			nn.ReLU(True),
		)

		self.model_mlp = nn.Sequential(
			nn.Linear(256*3*3, 1024, bias=True),
			nn.ReLU(True),
			nn.Linear(1024, 256, bias=True),
			nn.ReLU(True),
			nn.Linear(256, 68*2, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, image_encoding: TensorType['bs', 256, 16, 16]) -> TensorType['bs', 68*2]:
		i = self.model_cnn(image_encoding)
		i = i.flatten(start_dim=1)
		i = self.model_mlp(i)
		return i


class mlp_2_cnn_link(nn.Module):
	def __init__(self, params):
		"""
		input shape in 68*2
		output shape has to be 256x16x16
		"""
		super(mlp_2_cnn_link, self).__init__()
		norm_layer = get_norm_layer(norm_type=params['norm_layer'])
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.model_mlp = nn.Sequential(
			nn.Linear(68*2, 256, bias=True),
			nn.Sigmoid(),
			nn.Linear(256, 1024, bias=True),
			nn.ReLU(True),
			nn.Linear(1024, 256*3*3, bias=True),
			nn.ReLU(True),
		)

		self.model_cnn = nn.Sequential(
			# input is reshaped into 256x3x3
			nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=1, output_padding=0, bias=use_bias),
			# 256x7x7
			norm_layer(int(256)),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=1, output_padding=1, bias=use_bias),
			# 256x16x16
			norm_layer(int(256)),
			nn.ReLU(True),
			ResnetBlock(dim=256, padding_type='reflect', norm_layer=norm_layer,
			            use_dropout=params['dropout'], use_bias=use_bias)
		)

	def forward(self, y: TensorType['bs', 68*2]) -> TensorType['bs', 256, 16, 16]:
		i = self.model_mlp(y)
		i = i.reshape(-1, 256, 3, 3)
		i = self.model_cnn(i)
		return i


class define_Face2Face(nn.Module):
	def __init__(self, encoder, decoder):
		super(define_Face2Face, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


def get_cnn_mlp(params: dict):
	"""
	Return building block for the Face task:
	images are 3x68x68 and landmark are 2x68
	"""
	x_to_x = define_G(input_nc=params['in_dim'], output_nc=params['in_dim'], ngf=params['ngf'], netG=params['netG'],
	                  norm=params['norm_layer'], use_dropout=params['dropout'], init_type=params['init_type'],
	                  init_gain=params['init_gain'], o_activation=params['o_activation_x'])
	# after the encoder, the feature map is of shape 256x16x16
	face_encoder = mlp_2_cnn_link(params=params)
	face_decoder = cnn_2_mlp_link(params=params)
	y_to_y = define_Face2Face(encoder=face_encoder, decoder=face_decoder)

	link_x_to_y = get_cnn_link(params)
	link_y_to_x = get_cnn_link(params)

	return [x_to_x, y_to_y, link_x_to_y, link_y_to_x]
