from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from Utils.loss_metrics import iou_torch, NRMSE
from Utils.utils import seg2onehot
from src.Networks.Critic import Critic
from src.Networks.CycleGAN.networks import define_D
from src.Networks.IODA_SOP_utils import get_full_cnn_resnet, get_cnn_mlp, get_full_cnn_native, get_cnn_mlp_native

patch_typeguard()  # use before @typechecked


class FakeDiscriminator(nn.Module):
	def forward(self, x):
		batch_size = x.shape[0]
		return torch.full([batch_size, 1], 0.5, device=x.device)


class Masks:
	def __init__(self, mode):
		self.mask_x  = mode[:, 0]
		self.mask_y  = mode[:, 1]
		self.mask_xy = mode[:, 2]

		self.mask_only_x = self.mask_x * (1 - self.mask_xy)
		self.mask_only_y = self.mask_y * (1 - self.mask_xy)

		self.nb_x = self.mask_x.sum()
		self.nb_y = self.mask_y.sum()
		self.nb_xy = self.mask_xy.sum()
		self.nb_only_x = self.nb_x - self.nb_xy
		self.nb_only_y = self.nb_y - self.nb_xy

		self.mask_dict = {
			'x' : (self.mask_x, self.nb_x),
			'y' : (self.mask_y, self.nb_y),
			'xy': (self.mask_xy, self.nb_xy),

			'ox': (self.mask_only_x, self.nb_only_x),
			'oy': (self.mask_only_y, self.nb_only_y),
		}

	def get_mask(self, key: str):
		return self.mask_dict[key][0].reshape(-1, 1)

	def get_nb(self, key: str):
		return self.mask_dict[key][1]

	def __call__(self, tensor: torch.Tensor, key: str):
		batch_size = tensor.shape[0]
		if key is None:
			return torch.tensor(0.)
		if key == 'not':
			return tensor.mean()
		number = self.get_nb(key)
		if number == 0:
			return torch.tensor(0.)
		tensor = tensor.reshape(batch_size, -1).mean(dim=1)
		return (tensor.reshape(batch_size, -1) * self.get_mask(key)).sum() / number


@dataclass
class LSM_Output:
	x: torch.Tensor
	y: torch.Tensor

	latent_x: torch.Tensor
	latent_y: torch.Tensor

	xx: torch.Tensor
	yy: torch.Tensor

	latent_xy: torch.Tensor
	latent_yx: torch.Tensor

	xy: torch.Tensor
	yx: torch.Tensor
	###
	discriminator_decision_fake_y: torch.Tensor
	discriminator_decision_real_y: torch.Tensor

	discriminator_decision_fake_x: torch.Tensor
	discriminator_decision_real_x: torch.Tensor

	discriminator_decision_fake_xL: torch.Tensor
	discriminator_decision_real_xL: torch.Tensor

	discriminator_decision_fake_yL: torch.Tensor
	discriminator_decision_real_yL: torch.Tensor


@dataclass
class LSM_Lambdas:
		lambda_xx_loss: float
		lambda_yy_loss: float

		lambda_latent_supervised: float

		lambda_supervised_domain_y: float
		lambda_supervised_domain_x: float

		lambda_adv_g_domain: float
		lambda_adv_g_latent: float

		lambda_adv_d_domain: float
		lambda_adv_d_latent: float


class PL_LSM_Base(pl.LightningModule):
	def __init__(
			self,
			image_res: int,
			input_dimension: int,
			output_dimension: int,

			input_loss,
			output_loss,
			latent_loss,
			domain_adversarial_loss,
			latent_adversarial_loss,

			lambdas: LSM_Lambdas,

			lr: float,
			params: dict,
	):
		super().__init__()
		self.automatic_optimization = False

		self.image_res = image_res
		self.in_dim = input_dimension
		self.out_dim = output_dimension
		self.lr = lr
		self.params = params

		x_to_x, y_to_y, link_x_to_y, link_y_to_x = self.instantiate_network_generator(params=params)
		d_x, d_y, d_xL, d_yL = self.instantiate_network_discriminator(params=params)

		# region assign networks
		self.x_to_latent = x_to_x.encoder
		self.latent_to_x = x_to_x.decoder

		self.y_to_latent = y_to_y.encoder
		self.latent_to_y = y_to_y.decoder

		self.xL_to_yL = link_x_to_y
		self.yL_to_xL = link_y_to_x

		if lambdas.lambda_adv_g_domain == 0 and lambdas.lambda_adv_d_domain == 0:
			self.discriminator_x = FakeDiscriminator()
			self.discriminator_y = FakeDiscriminator()
		else:
			self.discriminator_x = d_x
			self.discriminator_y = d_y
		self.discriminator_xL = d_xL
		self.discriminator_yL = d_yL
		# endregion

		self.loss_input_supervised  = input_loss
		self.loss_output_supervised = output_loss
		self.loss_latent_supervised = latent_loss

		self.loss_adversarial_domain_space = domain_adversarial_loss
		self.loss_adversarial_latent_space = latent_adversarial_loss

		self.l = lambdas

	def instantiate_network_generator(self, params: dict):
		raise NotImplemented

	def instantiate_network_discriminator(self, params: dict):
		raise NotImplemented

	@typechecked
	def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> LSM_Output:
		x, y = batch

		# encoding part
		latent_x = self.x_to_latent(x)
		latent_y = self.y_to_latent(y)

		# auto encoder part
		xx = self.latent_to_x(latent_x)
		yy = self.latent_to_y(latent_y)

		# latent translation part
		latent_xy = self.xL_to_yL(latent_x)
		latent_yx = self.yL_to_xL(latent_y)

		# translation part
		xy = self.latent_to_y(latent_xy)
		yx = self.latent_to_x(latent_yx)

		# region discriminators decisions region
		discriminator_decision_fake_y = self.discriminator_y(xy)
		discriminator_decision_real_y = self.discriminator_y(y)

		discriminator_decision_fake_x = self.discriminator_x(yx)
		discriminator_decision_real_x = self.discriminator_x(x)

		discriminator_decision_fake_xL = self.discriminator_xL(latent_yx)
		discriminator_decision_real_xL = self.discriminator_xL(latent_x)

		discriminator_decision_fake_yL = self.discriminator_yL(latent_xy)
		discriminator_decision_real_yL = self.discriminator_yL(latent_y)
		# endregion

		return LSM_Output(
			x=x, y=y,
			xx=xx, yy=yy,

			latent_x=latent_x, latent_y=latent_y,
			latent_xy=latent_xy, latent_yx=latent_yx,

			xy=xy, yx=yx,
			###
			discriminator_decision_fake_y=discriminator_decision_fake_y,
			discriminator_decision_real_y = discriminator_decision_real_y,

			discriminator_decision_fake_x=discriminator_decision_fake_x,
			discriminator_decision_real_x=discriminator_decision_real_x,

			discriminator_decision_fake_xL=discriminator_decision_fake_xL,
			discriminator_decision_real_xL=discriminator_decision_real_xL,

			discriminator_decision_fake_yL=discriminator_decision_fake_yL,
			discriminator_decision_real_yL=discriminator_decision_real_yL,
		)

	def configure_optimizers(self):
		generator_opti = torch.optim.Adam([
			{'params': self.x_to_latent.parameters()},
			{'params': self.latent_to_x.parameters()},

			{'params': self.y_to_latent.parameters()},
			{'params': self.latent_to_y.parameters()},

			{'params': self.xL_to_yL.parameters()},
			{'params': self.yL_to_xL.parameters()},
		], lr=self.lr)

		discriminator_opti = torch.optim.Adam([
			{'params': self.discriminator_x.parameters()},
			{'params': self.discriminator_y.parameters()},
			{'params': self.discriminator_xL.parameters()},
			{'params': self.discriminator_yL.parameters()},
		], lr=self.lr)

		return discriminator_opti, generator_opti

	def __adversarial_loss_computation(self, fakes_decision, real_decision, loss_function, masks, key_fake, key_real):
		generator_fake_loss, (discriminator_fake_loss, discriminator_real_loss) = \
			loss_function.get_triplet(fakes_decision, real_decision)

		generator_fake_loss = masks(generator_fake_loss, key_fake)
		discriminator_fake_loss = masks(discriminator_fake_loss, key_fake)
		discriminator_real_loss = masks(discriminator_real_loss, key_real)

		generator_loss = generator_fake_loss
		discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) / 2

		return generator_loss, discriminator_loss

	def compute_metrics(self, pred: LSM_Output, masks: Masks, log_dict: dict, is_train: bool) -> dict:
		raise NotImplemented()

	def compute_logs(self, pred: LSM_Output, mode, log_dict: dict, is_train: bool) -> dict:
		masks = Masks(mode)

		# region compute loss
		# auto-encoder
		xx_loss = masks(self.loss_input_supervised(pred.xx, pred.x), 'x')
		yy_loss = masks(self.loss_output_supervised(pred.yy, pred.y), 'y')

		# latent translation with GT
		Lx_Lyx_loss = masks(self.loss_latent_supervised(pred.latent_yx, pred.latent_x), 'xy')
		Ly_Lxy_loss = masks(self.loss_latent_supervised(pred.latent_xy, pred.latent_y), 'xy')

		# translation with GT
		xy_loss = masks(self.loss_output_supervised(pred.xy, pred.y), 'xy')
		yx_loss = masks(self.loss_input_supervised(pred.yx, pred.x), 'xy')

		# region adversarial losses
		# Domain Space
		generator_domain_x_loss, discriminator_domain_x_loss = self.__adversarial_loss_computation(
			fakes_decision=pred.discriminator_decision_fake_x, real_decision=pred.discriminator_decision_real_x,
			loss_function=self.loss_adversarial_domain_space,
			masks=masks,
			key_fake='y', key_real='x',
		)

		generator_domain_y_loss, discriminator_domain_y_loss = self.__adversarial_loss_computation(
			fakes_decision=pred.discriminator_decision_fake_y, real_decision=pred.discriminator_decision_real_y,
			loss_function=self.loss_adversarial_domain_space,
			masks=masks,
			key_fake='x', key_real='y',
		)

		# Latent Space
		generator_domain_xL_loss, discriminator_domain_xL_loss = self.__adversarial_loss_computation(
			fakes_decision=pred.discriminator_decision_fake_xL, real_decision=pred.discriminator_decision_real_xL,
			loss_function=self.loss_adversarial_latent_space,
			masks=masks,
			key_fake='y', key_real='x',
		)

		generator_domain_yL_loss, discriminator_domain_yL_loss = self.__adversarial_loss_computation(
			fakes_decision=pred.discriminator_decision_fake_yL, real_decision=pred.discriminator_decision_real_yL,
			loss_function=self.loss_adversarial_latent_space,
			masks=masks,
			key_fake='x', key_real='y',
		)
		# endregion

		# region accumulates losses
		generator_loss_part = \
			+ xx_loss * self.l.lambda_xx_loss  \
			+ yy_loss * self.l.lambda_yy_loss \
			\
			+ Lx_Lyx_loss * self.l.lambda_latent_supervised \
			+ Ly_Lxy_loss * self.l.lambda_latent_supervised \
			\
			+ xy_loss * self.l.lambda_supervised_domain_y \
			+ yx_loss * self.l.lambda_supervised_domain_x \
			\
			+ generator_domain_x_loss * self.l.lambda_adv_g_domain \
			+ generator_domain_y_loss * self.l.lambda_adv_g_domain \
			\
			+ generator_domain_xL_loss * self.l.lambda_adv_g_latent \
			+ generator_domain_yL_loss * self.l.lambda_adv_g_latent

		discriminator_loss_part = \
			+ discriminator_domain_x_loss * self.l.lambda_adv_d_domain \
			+ discriminator_domain_y_loss * self.l.lambda_adv_d_domain \
			\
			+ discriminator_domain_xL_loss * self.l.lambda_adv_d_latent \
			+ discriminator_domain_yL_loss * self.l.lambda_adv_d_latent
		# endregion

		# endregion

		# region compute metrics
		log_dict |= self.compute_metrics(pred, masks, log_dict, is_train)
		# endregion

		return log_dict | {
			'loss/xx_loss': xx_loss,
			'loss/yy_loss': yy_loss,

			'loss/Lx_Lyx_loss': Lx_Lyx_loss,
			'loss/Ly_Lxy_loss': Ly_Lxy_loss,

			'loss/xy_loss': xy_loss,
			'loss/yx_loss': yx_loss,

			'loss/generator_domain_x_loss': generator_domain_x_loss,
			'loss/generator_domain_y_loss': generator_domain_y_loss,

			'loss/generator_domain_xL_loss': generator_domain_xL_loss,
			'loss/generator_domain_yL_loss': generator_domain_yL_loss,

			'loss/discriminator_domain_x_loss': discriminator_domain_x_loss,
			'loss/discriminator_domain_y_loss': discriminator_domain_y_loss,

			'loss/discriminator_domain_xL_loss': discriminator_domain_xL_loss,
			'loss/discriminator_domain_yL_loss': discriminator_domain_yL_loss,

			'loss/generator_loss_part': generator_loss_part,
			'loss/discriminator_loss_part': discriminator_loss_part,
		}

	def _step(self, batch, batch_idx, is_train: bool) -> dict:
		x, y, mode = batch
		pred: LSM_Output = self((x, y))
		log_dict = dict()
		log_dict = self.compute_logs(pred, mode, log_dict, is_train)

		return log_dict

	def training_step(self, batch, batch_idx):
		prefix = 'train'
		opt_discriminator, opt_generator = self.optimizers()

		# discriminator step
		log_dict = self._step(batch=batch, batch_idx=batch_idx, is_train=True)

		opt_discriminator.zero_grad()
		self.manual_backward(log_dict['loss/discriminator_loss_part'])
		opt_discriminator.step()

		# generator steop
		log_dict = self._step(batch=batch, batch_idx=batch_idx, is_train=True)

		opt_generator.zero_grad()
		self.manual_backward(log_dict['loss/generator_loss_part'])
		opt_generator.step()

		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)

	def validation_step(self, batch, batch_idx):
		prefix = 'valid'
		x, y = batch
		batch_size = x.shape[0]
		mode = torch.ones([batch_size, 3], device=self.device)

		log_dict = self._step(batch=(x, y, mode), batch_idx=batch_idx, is_train=False)
		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)

	def test_step(self, batch, batch_idx):
		prefix = 'test'
		x, y = batch
		batch_size = x.shape[0]
		mode = torch.ones([batch_size, 3], device=self.device)

		log_dict = self._step(batch=(x, y, mode), batch_idx=batch_idx, is_train=False)
		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)


class PL_LSM_FULL_CNN(PL_LSM_Base):
	def instantiate_network_generator(self, params: dict):
		"""
			(x_to_x, y_to_y, link_x_to_y, link_y_to_x)
		"""
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_full_cnn_native(params)
			return x_to_x, y_to_y, link_x_to_y, link_y_to_x
		else:
			return get_full_cnn_resnet(params)

	def instantiate_network_discriminator(self, params: dict):
		"""d_x, d_y, d_xL, d_yL"""
		d_x = define_D(input_nc=self.in_dim, ndf=params['ndf'], netD=params['netD_domain'], n_layers_D=params['n_layers_D_domain'],
		                norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])
		d_y = define_D(input_nc=self.out_dim, ndf=params['ndf'], netD=params['netD_domain'], n_layers_D=params['n_layers_D_domain'],
		               norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

		if params['use_native_archi']:
			d_xL = Critic(need_GRL=False, layers=params['native_discriminator_latent_layer_in'] , output_function=params['native_discriminator_latent_act_in'])
			d_yL = Critic(need_GRL=False, layers=params['native_discriminator_latent_layer_out'], output_function=params['native_discriminator_latent_act_out'])
		else:
			d_xL = define_D(input_nc=params['latent_resolution'], ndf=params['ndf'], netD=params['netD_latent'], n_layers_D=params['n_layers_D_latent'],
			               norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

			d_yL = define_D(input_nc=params['latent_resolution'], ndf=params['ndf'], netD=params['netD_latent'],
			                n_layers_D=params['n_layers_D_latent'],
			                norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

		return d_x, d_y, d_xL, d_yL

	def compute_metrics(self, pred: LSM_Output, masks: Masks, log_dict: dict, is_train: bool) -> dict:
		miou_xy = masks(iou_torch((pred.xy > 0.5).int(), pred.y.int(), reduction=False), key='xy' if is_train else 'not')
		miou_yy = masks(iou_torch((pred.yy > 0.5).int(), pred.y.int(), reduction=False), key='y' if is_train else 'not')

		mse_yx = masks(F.mse_loss(pred.yx, pred.x, reduction='none'), key='xy' if is_train else 'not')
		mse_xx = masks(F.mse_loss(pred.xx, pred.x, reduction='none'), key='x' if is_train else 'not')
		l1_yx  = masks(F.l1_loss(pred.yx, pred.x, reduction='none'), key='xy' if is_train else 'not')
		l1_xx  = masks(F.l1_loss(pred.xx, pred.x, reduction='none'), key='x' if is_train else 'not')

		return log_dict | {
			'metric/miou_xy': miou_xy,
			'metric/miou_yy': miou_yy,

			'metric/mse_yx': mse_yx,
			'metric/mse_xx': mse_xx,

			'metric/l1_yx': l1_yx,
			'metric/l1_xx': l1_xx,
		}

	def training_step(self, batch, batch_idx):
		x, y, mode = batch
		y = seg2onehot(y, nb_classes=self.out_dim)
		return super().training_step((x, y, mode), batch_idx=batch_idx)

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y = seg2onehot(y, nb_classes=self.out_dim)
		super().validation_step((x, y), batch_idx=batch_idx)

	def test_step(self, batch, batch_idx):
		x, y = batch
		y = seg2onehot(y, nb_classes=self.out_dim)
		super().test_step((x, y), batch_idx=batch_idx)


class PL_LSM_CNN_MLP(PL_LSM_Base):
	"""
		x_to_x, y_to_y, link_x_to_y, link_y_to_x = self.instantiate_network_generator(params=params)
		d_x, d_y, d_xL, d_yL = self.instantiate_network_discriminator(params=params)
	"""
	def instantiate_network_generator(self, params: dict) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
		"""
			(x_to_x, y_to_y, link_x_to_y, link_y_to_x)
		"""
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp_native(params)
			return x_to_x, y_to_y, link_x_to_y, link_y_to_x
		else:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp(params)
			return x_to_x, y_to_y, link_x_to_y, link_y_to_x

	def instantiate_network_discriminator(self, params: dict):
		"""d_x, d_y, d_xL, d_yL"""
		d_x = define_D(input_nc=self.in_dim, ndf=params['ndf'], netD=params['netD_domain'], n_layers_D=params['n_layers_D_domain'],
		               norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

		d_y = Critic(
			need_GRL=False,
			layers=params['landmark_d_layers'],
			output_function=params['landmark_d_act'],
		)

		if params['use_native_archi']:
			d_xL = Critic(need_GRL=False, layers=params['native_discriminator_latent_layer_in'] , output_function=params['native_discriminator_latent_act_in'])
			d_yL = Critic(need_GRL=False, layers=params['native_discriminator_latent_layer_out'], output_function=params['native_discriminator_latent_act_out'])
		else:
			d_xL = define_D(input_nc=params['latent_resolution'], ndf=params['ndf'], netD=params['netD_latent'], n_layers_D=params['n_layers_D_latent'],
			                norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

			d_yL = define_D(input_nc=params['latent_resolution'], ndf=params['ndf'], netD=params['netD_latent'], n_layers_D=params['n_layers_D_latent'],
			                norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

		return d_x, d_y, d_xL, d_yL

	def compute_metrics(self, pred: LSM_Output, masks: Masks, log_dict: dict, is_train: bool) -> dict:
		nrmse_xy = masks(NRMSE(pred.xy, pred.y, reduction=False), key='xy' if is_train else 'not')
		nrmse_yy = masks(NRMSE(pred.yy, pred.y, reduction=False), key='y'  if is_train else 'not')

		mse_yx = masks(F.mse_loss(pred.yx, pred.x, reduction='none'), key='xy' if is_train else 'not')
		mse_xx = masks(F.mse_loss(pred.xx, pred.x, reduction='none'), key='x'  if is_train else 'not')
		l1_yx  = masks(F.l1_loss(pred.yx, pred.x , reduction='none'), key='xy' if is_train else 'not')
		l1_xx  = masks(F.l1_loss(pred.xx, pred.x , reduction='none'), key='x'  if is_train else 'not')

		return log_dict | {
			'metric/nrmse_xy': nrmse_xy,
			'metric/nrmse_yy': nrmse_yy,

			'metric/mse_yx': mse_yx,
			'metric/mse_xx': mse_xx,

			'metric/l1_yx': l1_yx,
			'metric/l1_xx': l1_xx,
		}
