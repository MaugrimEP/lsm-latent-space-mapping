from __future__ import annotations

from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType

from Utils.loss_metrics import NRMSE
from src.Model_CycleGAN_General import PL_CycleGAN_Base, CycleGAN_Output, CycleGAN_Losses
from src.Networks.Critic import Critic
from src.Networks.CycleGAN.networks import define_D
from src.Networks.IODA_SOP_utils import get_cnn_mlp, get_cnn_mlp_native

type_img = TensorType['bs', 'c', 'h', 'w']
type_vec = TensorType['bs', 'vector_dim']


class CycleGAN_CNN_MLP(PL_CycleGAN_Base):

	def _specific_step(self, pred: CycleGAN_Output, losses: CycleGAN_Losses, log_dict: dict, prefix: str) -> dict:
		NRMSE_xy = NRMSE(pred.xy, pred.y)
		NRMSE_yy = NRMSE(pred.yy, pred.y)

		mse_xy = F.mse_loss(pred.xy, pred.y)
		mse_yy = F.mse_loss(pred.yy, pred.y)

		log_dict |= {
			'NRMSE_xy': NRMSE_xy,
			'NRMSE_yy': NRMSE_yy,
			'mse_xy': mse_xy,
			'mse_yy': mse_yy,
		}
		return log_dict

	def get_x_to_y(self, params) -> nn.Module:
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp_native(params)
		else:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp(params)

		x_to_y = nn.Sequential(
			x_to_x.encoder,
			link_x_to_y,
			y_to_y.decoder,
		)
		return x_to_y

	def get_y_to_x(self, params) -> nn.Module:
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp_native(params)
		else:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp(params)

		y_to_x = nn.Sequential(
			y_to_y.encoder,
			link_y_to_x,
			x_to_x.decoder,
		)
		return y_to_x

	def get_discriminator_x(self, params) -> nn.Module:
		return define_D(input_nc=self.in_dim, ndf=params['ndf'], netD=params['netD'], n_layers_D=params['n_layers_D'],
		                norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

	def get_discriminator_y(self, params) -> nn.Module:
		return Critic(
			need_GRL=False,
			layers=params['landmark_d_layers'],
			output_function=params['landmark_d_act'],
		)
