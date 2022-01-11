import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType

from Utils.loss_metrics import iou_torch, NRMSE
from src.Model_Pix2Pix_General import PL_Pix2Pix_NBase, Pix2Pix_Output, Pix2Pix_Losses
from src.Networks.CycleGAN.networks import define_G, define_D, _get_activation
from src.Networks.IODA_SOP_utils import full_cnn_network, _get_linking_part, _get_mlp_network, \
	_get_mlp_linking_part, get_cnn_mlp, get_cnn_mlp_native


class Augment(nn.Module):
	"""We have the image, and the point face, we need to augment to the image size and concat it"""

	def __init__(self, out_dim: int):
		super(Augment, self).__init__()
		self.out_dim = out_dim

	def forward(self, x: TensorType['bs', 3, 'h', 'w'], y: TensorType['bs', 68*2]):
		bs, c, h, w = x.shape
		y_image = y.repeat(h, w, 1, 1)  # broadcast the landmark on each pixel
		# y_image.shape = [h, w, bs, 68*2]
		y_image = y_image.permute(2, 3, 0, 1)  # put back the 68*2 as the channels, and the batch first
		return torch.cat([x, y_image], dim=1)


class PL_Pix2Pix_CNN_MLP(PL_Pix2Pix_NBase):

	def get_generator(self, params: dict) -> nn.Module:
		"""Takes X: image and produce Y: face point"""
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp_native(params)
		else:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp(params)

		x_to_y = nn.Sequential()
		x_to_y.add_module('encoder'   , x_to_x.encoder)
		x_to_y.add_module('link'      , link_x_to_y)
		x_to_y.add_module('decoder'   , y_to_y.decoder)

		return x_to_y

	def get_discriminator(self, params: dict) -> nn.Module:
		return define_D(
			input_nc=params['discriminator_input_channels'], ndf=params['ndf'], netD=params['netD'], n_layers_D=params['n_layers_D'],
		    norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'],
		)

	def get_link_to_augment_g(self, params: dict) -> nn.Module:
		return nn.Identity()

	def get_link_to_augment_d(self, params: dict) -> nn.Module:
		return Augment(self.out_dim)

	def _specific_step(self, pred: Pix2Pix_Output, losses: Pix2Pix_Losses, log_dict: dict, prefix: str) -> dict:
		bs, c, h, w = pred.x.shape

		NRMSE_xy = NRMSE(pred.fake_y, pred.y)

		mse_xy = F.mse_loss(pred.fake_y, pred.y)

		log_dict |= {
			'NRMSE_xy': NRMSE_xy,
			'mse_xy': mse_xy,
		}

		return log_dict
