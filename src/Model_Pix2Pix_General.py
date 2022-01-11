from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from Utils.loss_metrics import _get_adversarial_loss, Adversarial_loss
from src.Networks.CycleGAN.networks import get_scheduler

patch_typeguard()  # use before @typechecked


@dataclass
class Pix2Pix_Output:
	x: torch.Tensor
	y: torch.Tensor

	fake_y: torch.Tensor

	real_decision: torch.Tensor
	fake_decision: torch.Tensor


@dataclass
class Pix2Pix_Losses:
	discriminator_real_loss: torch.Tensor
	discriminator_fake_loss: torch.Tensor

	generator_fake_loss      : torch.Tensor
	generator_supervised_loss: torch.Tensor

	discriminator_loss : torch.Tensor
	generator_loss     : torch.Tensor

	@staticmethod
	def compute_elem_wise(pred: Pix2Pix_Output, adversial_loss: Adversarial_loss, supervised_loss_f, lamda_supervised) -> Pix2Pix_Losses:
		batch_size, c, h, w = pred.x.shape

		# region loss computation
		discriminator_fake_loss = adversial_loss.d_fakes(pred.fake_decision).mean()
		discriminator_real_loss = adversial_loss.d_reals(pred.real_decision).mean()

		generator_fake_loss       = adversial_loss.g_fake(pred.fake_decision).mean()
		generator_supervised_loss = supervised_loss_f(pred.fake_y, pred.y).mean()

		discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) * 0.5
		generator_loss     = generator_fake_loss + generator_supervised_loss * lamda_supervised

		return Pix2Pix_Losses(
			discriminator_real_loss  =discriminator_real_loss,
			discriminator_fake_loss  =discriminator_fake_loss,

			generator_fake_loss      =generator_fake_loss,
			generator_supervised_loss=generator_supervised_loss,

			discriminator_loss=discriminator_loss,
			generator_loss    =generator_loss,
		)

	def add_to_dict(self, log_dict: dict) -> dict:
		return log_dict | dict(
			discriminator_fake_loss=self.discriminator_fake_loss,
			discriminator_real_loss=self.discriminator_real_loss,

			generator_fake_loss      =self.generator_fake_loss,
			generator_supervised_loss=self.generator_supervised_loss,

			discriminator_loss=self.discriminator_loss,
			generator_loss    =self.generator_loss,
		)


class PL_Pix2Pix_NBase(pl.LightningModule):
	def __init__(
			self,
			image_res: int,
			input_dimension: int,
			output_dimension: int,
			lr: float,
			params: dict,
			output_loss,
			lambda_supervised: float,
	):
		super().__init__()
		self.automatic_optimization = False

		self.image_res = image_res
		self.in_dim = input_dimension
		self.out_dim = output_dimension

		self.lr = lr

		self.params = params

		self.adversarial_loss = _get_adversarial_loss(params['adversarial_loss'])
		self.output_loss      = output_loss

		self.augment_generator = self.get_link_to_augment_g(params)
		self.generator = self.get_generator(params)

		self.augment_discriminator = self.get_link_to_augment_d(params)
		self.discriminator = self.get_discriminator(params)

		self.lambda_supervised = lambda_supervised

	# region not implemented
	def get_generator(self, params: dict) -> nn.Module:
		raise NotImplemented

	def get_discriminator(self, params: dict) -> nn.Module:
		raise NotImplemented

	def get_link_to_augment_g(self, params: dict) -> nn.Module:
		raise NotImplemented

	def get_link_to_augment_d(self, params: dict) -> nn.Module:
		raise NotImplemented

	# endregion

	@typechecked
	def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Pix2Pix_Output:
		x, y = batch

		augment_g = self.augment_generator(x)
		fake_y = self.generator(augment_g)

		real_pair = self.augment_discriminator(x, y)
		fake_pair = self.augment_discriminator(x, fake_y)

		real_decision = self.discriminator(real_pair)
		fake_decision = self.discriminator(fake_pair)

		return Pix2Pix_Output(x=x, y=y, fake_y=fake_y, real_decision=real_decision, fake_decision=fake_decision)

	def configure_optimizers(self):
		generator_opti = torch.optim.Adam([
			{'params': self.augment_generator.parameters()},
			{'params': self.generator.parameters()},
		], lr=self.lr, betas=(self.params['beta1'], 0.999))

		discriminator_opti = torch.optim.Adam([
			{'params': self.augment_discriminator.parameters()},
			{'params': self.discriminator.parameters()},
		], lr=self.lr, betas=(self.params['beta1'], 0.999))

		if self.params['use_scheduler']:
			scheduler_generator     = get_scheduler(generator_opti    , self.params)
			scheduler_discriminator = get_scheduler(discriminator_opti, self.params)
			return [discriminator_opti, generator_opti], [scheduler_discriminator, scheduler_generator]
		else:
			return discriminator_opti, generator_opti

	def _global_step(self, batch, batch_idx, prefix) -> Tuple[Pix2Pix_Output, Pix2Pix_Losses, dict]:
		x, y = batch
		pred: Pix2Pix_Output = self((x, y))

		log_dict = dict()

		losses = Pix2Pix_Losses.compute_elem_wise(
			pred=pred,
			adversial_loss=self.adversarial_loss,
			supervised_loss_f=self.output_loss,
			lamda_supervised=self.lambda_supervised,
		)
		log_dict = losses.add_to_dict(log_dict=log_dict)
		log_dict |= {
			f'{prefix}/batch_idx': batch_idx,
		}

		return pred, losses, log_dict

	def _specific_step(self, pred: Pix2Pix_Output, losses: Pix2Pix_Losses, log_dict: dict, prefix: str) -> dict:
		return log_dict

	def training_step(self, batch, batch_idx):
		prefix = 'train'
		opt_discriminator, opt_generator = self.optimizers()

		# discriminator step
		pred, losses, log_dict = self._global_step(batch=batch, batch_idx=batch_idx, prefix=prefix)
		log_dict = self._specific_step(pred=pred, losses=losses, log_dict=log_dict, prefix=prefix)

		opt_discriminator.zero_grad()
		self.manual_backward(log_dict['discriminator_loss'])
		opt_discriminator.step()

		# generator steop
		pred, losses, log_dict = self._global_step(batch=batch, batch_idx=batch_idx, prefix=prefix)
		log_dict = self._specific_step(pred=pred, losses=losses, log_dict=log_dict, prefix=prefix)

		opt_generator.zero_grad()
		self.manual_backward(log_dict['generator_loss'])
		opt_generator.step()

		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)

	def validation_step(self, batch, batch_idx):
		prefix = "valid"
		pred, losses, log_dict = self._global_step(batch=batch, batch_idx=batch_idx, prefix=prefix)
		log_dict = self._specific_step(pred=pred, losses=losses, log_dict=log_dict, prefix=prefix)
		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)

	def test_step(self, batch, batch_idx):
		prefix = "test"
		pred, losses, log_dict = self._global_step(batch=batch, batch_idx=batch_idx, prefix=prefix)
		log_dict = self._specific_step(pred=pred, losses=losses, log_dict=log_dict, prefix=prefix)
		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{key}', value=value)
