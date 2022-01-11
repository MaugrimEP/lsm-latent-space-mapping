from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from Utils.loss_metrics import _get_adversarial_loss, Adversarial_loss
from src.Networks.CycleGAN.networks import get_scheduler

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()  # use before @typechecked


type_x = torch.Tensor
type_y = torch.Tensor


@dataclass
class CycleGAN_Output:
	x: type_x
	y: type_y

	xy: type_y
	yx: type_x

	xx: type_x
	yy: type_y

	d_x: TensorType['bs', 'nb_patch_x']
	d_y: TensorType['bs', 'nb_patch_y']

	d_xy: TensorType['bs', 'nb_patch_y']
	d_yx: TensorType['bs', 'nb_patch_x']


@dataclass
class CycleGAN_Losses:
	discriminator_real_x: torch.Tensor
	discriminator_real_y: torch.Tensor

	discriminator_fake_xy: torch.Tensor
	discriminator_fake_yx: torch.Tensor

	generator_fake_xy: torch.Tensor
	generator_fake_yx: torch.Tensor

	cycle_xx: torch.Tensor
	cycle_yy: torch.Tensor

	@staticmethod
	def compute_elem_wise(
			pred: CycleGAN_Output, adversial_loss: Adversarial_loss,
			cycle_func_x, cycle_func_y,
	) -> CycleGAN_Losses:
		batch_size = pred.x.shape[0]

		# region loss computation
		discriminator_fake_loss_yx = adversial_loss.d_fakes(pred.d_yx).reshape(batch_size, -1).mean(dim=1)
		discriminator_fake_loss_xy = adversial_loss.d_fakes(pred.d_xy).reshape(batch_size, -1).mean(dim=1)

		generator_fake_loss_yx = adversial_loss.g_fake(pred.d_yx).reshape(batch_size, -1).mean(dim=1)
		generator_fake_loss_xy = adversial_loss.g_fake(pred.d_xy).reshape(batch_size, -1).mean(dim=1)

		discriminator_real_loss_x = adversial_loss.d_reals(pred.d_x).reshape(batch_size, -1).mean(dim=1)
		discriminator_real_loss_y = adversial_loss.d_reals(pred.d_y).reshape(batch_size, -1).mean(dim=1)

		cycle_xx = cycle_func_x(pred.xx, pred.x).reshape(batch_size, -1).mean(dim=1)
		cycle_yy = cycle_func_y(pred.yy, pred.y).reshape(batch_size, -1).mean(dim=1)

		return CycleGAN_Losses(
			discriminator_real_x=discriminator_real_loss_x,
			discriminator_real_y=discriminator_real_loss_y,

			discriminator_fake_xy=discriminator_fake_loss_xy,
			discriminator_fake_yx=discriminator_fake_loss_yx,

			generator_fake_xy=generator_fake_loss_xy,
			generator_fake_yx=generator_fake_loss_yx,

			cycle_xx=cycle_xx,
			cycle_yy=cycle_yy,
		)

	def aggregate(self, model: PL_CycleGAN_Base, log_dict: dict, mode: torch.Tensor) -> dict:
		mask_x = mode[:, 0]
		mask_y = mode[:, 1]

		nb_x = mask_x.sum()
		nb_y = mask_y.sum()

		loss_generator = 0.
		loss_discriminator = 0.
		if nb_x != 0:
			discriminator_real_x  = (self.discriminator_real_x  * mask_x).sum() / nb_x
			discriminator_fake_xy = (self.discriminator_fake_xy * mask_x).sum() / nb_x
			generator_fake_xy     = (self.generator_fake_xy     * mask_x).sum() / nb_x
			cycle_xx              = (self.cycle_xx              * mask_x).sum() / nb_x

			loss_generator += generator_fake_xy + cycle_xx * model.lambda_cycle
			loss_discriminator += (discriminator_real_x + discriminator_fake_xy) * 0.5

			log_dict |= {
				'discriminator_real_x' : discriminator_real_x,
				'discriminator_fake_xy': discriminator_fake_xy,
				'generator_fake_xy'    : generator_fake_xy,
				'cycle_xx'             : cycle_xx,
			}

		if nb_y != 0:
			discriminator_real_y  = (self.discriminator_real_y  * mask_y).sum() / nb_y
			discriminator_fake_yx = (self.discriminator_fake_yx * mask_y).sum() / nb_y
			generator_fake_yx     = (self.generator_fake_yx     * mask_y).sum() / nb_y
			cycle_yy              = (self.cycle_yy              * mask_y).sum() / nb_y

			loss_generator += generator_fake_yx + cycle_yy * model.lambda_cycle
			loss_discriminator += (discriminator_real_y + discriminator_fake_yx) * 0.5

			log_dict |= {
				'nb_x': nb_x,
				'nb_y': nb_y,

				'discriminator_real_y' : discriminator_real_y,
				'discriminator_fake_yx': discriminator_fake_yx,
				'generator_fake_yx'    : generator_fake_yx,
				'cycle_yy'             : cycle_yy,
			}

		log_dict |= {'loss_generator': loss_generator, 'loss_discriminator': loss_discriminator}

		return log_dict


class PL_CycleGAN_Base(pl.LightningModule):
	def __init__(
			self,
			image_res: int,
			input_dimension: int,
			output_dimension: int,
			lr_g: float,
			lr_d: float,
			lambda_cycle: float,
			params: dict,
			adversarial_loss,
			input_loss,
			output_loss,
	):
		super().__init__()
		self.automatic_optimization = False

		self.image_res = image_res
		self.in_dim = input_dimension
		self.out_dim = output_dimension

		self.lr_g = lr_g
		self.lr_d = lr_d

		self.lambda_cycle = lambda_cycle
		self.params = params

		self.adversarial_loss = adversarial_loss
		self.input_loss       = input_loss
		self.output_loss      = output_loss

		self.x_to_y = self.get_x_to_y(params=params)
		self.y_to_x = self.get_y_to_x(params=params)
		self.discriminatorX = self.get_discriminator_x(params)
		self.discriminatorY = self.get_discriminator_y(params)

	# region not implemented
	def get_x_to_y(self, params) -> nn.Module:
		raise NotImplemented

	def get_y_to_x(self, params) -> nn.Module:
		raise NotImplemented

	def get_discriminator_x(self, params) -> nn.Module:
		raise NotImplemented

	def get_discriminator_y(self, params) -> nn.Module:
		raise NotImplemented
	# endregion

	@typechecked
	def forward(self, batch: Tuple[type_x, type_y]) -> CycleGAN_Output:
		x, y = batch

		bs, c1, h, w = x.shape

		xy = self.x_to_y(x)
		yx = self.y_to_x(y)

		cycle_x = self.y_to_x(xy)
		cycle_y = self.x_to_y(yx)

		d_xy = self.discriminatorY(xy).reshape(bs, -1)
		d_yx = self.discriminatorX(yx).reshape(bs, -1)

		d_x = self.discriminatorX(x).reshape(bs, -1)
		d_y = self.discriminatorY(y).reshape(bs, -1)

		return CycleGAN_Output(
			x=x, y=y,
			xy=xy, yx=yx,
			xx=cycle_x, yy=cycle_y,
			d_x=d_x, d_y=d_y,
			d_xy=d_xy, d_yx=d_yx,
		)

	def configure_optimizers(self):
		generator_opti = torch.optim.Adam([
			{'params': self.x_to_y.parameters()},
			{'params': self.y_to_x.parameters()},
		], lr=self.lr_g, betas=(self.params['beta1'], 0.999))

		discriminator_opti = torch.optim.Adam([
			{'params': self.discriminatorX.parameters()},
			{'params': self.discriminatorY.parameters()},
		], lr=self.lr_d, betas=(self.params['beta1'], 0.999))

		if self.params['use_scheduler']:
			scheduler_generator     = get_scheduler(generator_opti    , self.params)
			scheduler_discriminator = get_scheduler(discriminator_opti, self.params)
			return [discriminator_opti, generator_opti], [scheduler_discriminator, scheduler_generator]
		else:
			return discriminator_opti, generator_opti

	def _global_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx, prefix) -> Tuple[CycleGAN_Output, CycleGAN_Losses, dict]:
		x, y, mode = batch
		log_dict = dict()

		pred: CycleGAN_Output = self((x, y))
		losses = CycleGAN_Losses.compute_elem_wise(
			pred=pred,
			adversial_loss=self.adversarial_loss,
			cycle_func_x=self.input_loss,
			cycle_func_y=self.output_loss,
		)
		log_dict = losses.aggregate(model=self, log_dict=log_dict, mode=mode)

		return pred, losses, log_dict

	def _specific_step(self, pred: CycleGAN_Output, losses: CycleGAN_Losses, log_dict: dict, prefix: str) -> dict:
		return log_dict

	def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
		prefix = 'train'
		opt_discriminator, opt_generator = self.optimizers()

		# discriminator step
		pred, losses, log_dict = self._global_step(batch=batch, batch_idx=batch_idx, prefix=prefix)
		with torch.no_grad():
			log_dict = self._specific_step(pred=pred, losses=losses, log_dict=log_dict, prefix=prefix)

		opt_discriminator.zero_grad()
		self.manual_backward(log_dict['loss_discriminator'])
		opt_discriminator.step()

		# generator steop
		pred, losses, log_dict = self._global_step(batch=batch, batch_idx=batch_idx, prefix=prefix)
		log_dict = self._specific_step(pred=pred, losses=losses, log_dict=log_dict, prefix=prefix)

		opt_generator.zero_grad()
		self.manual_backward(log_dict['loss_generator'])
		opt_generator.step()

		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)

	def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
		prefix = "valid"
		x, y = batch
		batch_size = x.shape[0]
		mode = torch.ones([batch_size, 2], device=self.device)

		pred, losses, log_dict = self._global_step(batch=(x, y, mode), batch_idx=batch_idx, prefix=prefix)
		log_dict = self._specific_step(pred=pred, losses=losses, log_dict=log_dict, prefix=prefix)

		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)

	def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
		prefix = "test"
		x, y = batch
		batch_size = x.shape[0]
		mode = torch.ones([batch_size, 2], device=self.device)

		pred, losses, log_dict = self._global_step(batch=(x, y, mode), batch_idx=batch_idx, prefix=prefix)
		log_dict = self._specific_step(pred=pred, losses=losses, log_dict=log_dict, prefix=prefix)

		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)
