import torch
from torch import nn
import torch.nn.functional as F

from Utils.loss_metrics import iou_torch
from Utils.utils import seg2onehot
from src.Model_Pix2Pix_General import PL_Pix2Pix_NBase, Pix2Pix_Output, Pix2Pix_Losses
from src.Networks.CycleGAN.networks import define_G, define_D, _get_activation
from src.Networks.IODA_SOP_utils import full_cnn_network, _get_linking_part, get_cnn_link, get_full_cnn_native


class Augment(nn.Module):
	def __init__(self, out_dim: int):
		super(Augment, self).__init__()
		self.out_dim = out_dim

	def forward(self, x, y):
		return torch.cat([x, y], dim=1)


class PL_Pix2Pix_FULL_CNN(PL_Pix2Pix_NBase):

	def get_generator(self, params: dict) -> nn.Module:
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_full_cnn_native(params)
			x_to_y = nn.Sequential()
			x_to_y.add_module('encoder', x_to_x.encoder)
			x_to_y.add_module('link'   , link_x_to_y)
			x_to_y.add_module('decoder', y_to_y.decoder)

			return x_to_y
		else:
			generator = define_G(
				input_nc=self.in_dim, output_nc=self.out_dim, ngf=params['ngf'], netG=params['netG'],
				norm=params['norm_layer'], use_dropout=params['dropout'], init_type=params['init_type'],
				init_gain=params['init_gain'], o_activation=params['o_activation'],
			)
			link_x_to_y = get_cnn_link(params)

			link = link_x_to_y if params['use_link'] else nn.Identity()

			x_to_y = nn.Sequential()
			x_to_y.add_module('encoder', generator.encoder)
			x_to_y.add_module('link'   , link)
			x_to_y.add_module('decoder', generator.decoder)

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
		miou_xy = iou_torch((pred.fake_y > 0.5).int(), pred.y.int())

		bs, c, h , w = pred.x.shape

		weights = (h * w) / torch.sum(pred.y, dim=[0, 2, 3])
		ce_xy = F.cross_entropy(pred.fake_y, pred.y, weight=weights)

		log_dict |= {'miou_xy': miou_xy, 'ce_xy': ce_xy}

		return log_dict

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		y = seg2onehot(y, self.out_dim)
		return super().training_step(batch=[x, y], batch_idx=batch_idx)

	def validation_step(self, train_batch, batch_idx):
		x, y = train_batch
		y = seg2onehot(y, self.out_dim)
		super().validation_step([x, y], batch_idx=batch_idx)

	def test_step(self, train_batch, batch_idx):
		x, y = train_batch
		y = seg2onehot(y, self.out_dim)
		super().test_step([x, y], batch_idx=batch_idx)
