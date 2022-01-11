from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable
import torch
from torch import nn
import pytorch_lightning as pl

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from Utils.utils import seg2onehot
from src.Networks.IODA_SOP_utils import get_full_cnn_resnet, get_cnn_mlp, get_full_cnn_native, get_cnn_mlp_native

patch_typeguard()  # use before @typechecked


@dataclass
class IODA_Output:
	x: torch.Tensor
	y: torch.Tensor

	xx: torch.Tensor
	yy: torch.Tensor
	xy: torch.Tensor


@dataclass
class IODA_Logs:
	xx_loss: torch.Tensor
	yy_loss: torch.Tensor
	xy_loss :torch.Tensor

	xx_metric: torch.Tensor
	yy_metric: torch.Tensor
	xy_metric: torch.Tensor


class PL_IODA_Base(pl.LightningModule):
	def __init__(
			self,
			image_res: int,
			input_dimension: int,
			output_dimension: int,

			input_metric,
			output_metric,
			full_metric,

			input_loss,
			output_loss,
			full_loss,

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
		x_to_x, y_to_y, link_x_to_y, _ = self.instantiate_network(params=params)
		self.x_to_x = x_to_x
		self.y_to_y = y_to_y
		self.x_to_y = nn.Sequential(x_to_x.encoder, link_x_to_y, y_to_y.decoder)
		self.mode = None

		self.input_metric  = input_metric
		self.output_metric = output_metric
		self.full_metric   = full_metric

		self.input_loss  = input_loss
		self.output_loss = output_loss
		self.full_loss   = full_loss

	def instantiate_network(self, params: dict) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
		raise NotImplemented

	def set_mode(self, mode: str):
		"""
		set the process for the current IODA Step
		:param mode: [ input | output | full ]
		:return:
		"""
		if mode not in ['input', 'output', 'full']:
			raise Exception(f'Unknown step: {mode=}')
		self.mode = mode

	@typechecked
	def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> IODA_Output:
		x, y = batch

		xx = self.x_to_x(x)
		yy = self.y_to_y(y)
		xy = self.x_to_y(x)

		return IODA_Output(x=x, y=y, xx=xx, yy=yy, xy=xy)

	def configure_optimizers(self):
		opt_input  = torch.optim.Adam([{'params': self.x_to_x.parameters()}], lr=self.lr)
		opt_output = torch.optim.Adam([{'params': self.y_to_y.parameters()}], lr=self.lr)
		opt_full   = torch.optim.Adam([{'params': self.x_to_y.parameters()}], lr=self.lr)
		return opt_input, opt_output, opt_full

	def compute_logs(self, pred: IODA_Output) -> IODA_Logs:
		xx_loss = self.input_loss(pred.xx, pred.x)
		yy_loss = self.output_loss(pred.yy, pred.y)
		xy_loss = self.full_loss(pred.xy, pred.y)

		with torch.no_grad():
			xy_metric = self.full_metric(pred.xy, pred.y)
			xx_metric = self.input_metric(pred.xx, pred.x)
			yy_metric = self.output_metric(pred.yy, pred.y)

		return IODA_Logs(
			xx_loss=xx_loss, yy_loss=yy_loss, xy_loss=xy_loss,
			xx_metric=xx_metric, yy_metric=yy_metric, xy_metric=xy_metric,
		)

	def _step(self, batch, batch_idx) -> dict:
		x, y = batch
		pred: IODA_Output = self((x, y))

		logs = self.compute_logs(pred)
		log_dict = {
			f'loss/{self.input_loss.name}_xx' : logs.xx_loss,
			f'loss/{self.output_loss.name}_yy': logs.yy_loss,
			f'loss/{self.full_loss.name}_xy'  : logs.xy_loss,

			f'metric/{self.input_metric.name}_xx' : logs.xx_metric,
			f'metric/{self.output_metric.name}_yy': logs.yy_metric,
			f'metric/{self.full_metric.name}_xy'  : logs.xy_metric,
		}

		curr_loss_value, curr_metric_value = {
			'input' : (logs.xx_loss, logs.xx_metric),
			'output': (logs.yy_loss, logs.yy_metric),
			'full'  : (logs.xy_loss, logs.xy_metric),
		}[self.mode]

		log_dict |= {
			f'batch_idx': batch_idx,
			f'loss'     : curr_loss_value,
		}
		return log_dict

	def training_step(self, batch, batch_idx):
		input_optimizer, output_optimizer, full_optimizer = self.optimizers()
		optimizer = {
			'input' : input_optimizer,
			'output': output_optimizer,
			'full'  : full_optimizer,
		}[self.mode]

		prefix = 'train'
		log_dict = self._step(batch=batch, batch_idx=batch_idx)

		optimizer.zero_grad()
		self.manual_backward(log_dict['loss'])
		optimizer.step()

		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{self.mode}/{key}', value=value)

	def validation_step(self, batch, batch_idx):
		prefix = 'valid'
		log_dict = self._step(batch=batch, batch_idx=batch_idx)
		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{self.mode}/{key}', value=value)

	def test_step(self, batch, batch_idx):
		prefix = 'test'
		log_dict = self._step(batch=batch, batch_idx=batch_idx)
		for key, value in log_dict.items():
			self.log(name=f'{prefix}/{self.mode}/{key}', value=value)


class PL_IODA_FULL_CNN(PL_IODA_Base):
	def instantiate_network(self, params: dict) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
		"""
			(x_to_x, y_to_y, link_x_to_y, link_y_to_x)
		"""
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_full_cnn_native(params)
			return x_to_x, y_to_y, link_x_to_y, link_y_to_x
		else:
			return get_full_cnn_resnet(params)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y = seg2onehot(y, nb_classes=self.out_dim)
		return super().training_step((x, y), batch_idx=batch_idx)

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y = seg2onehot(y, nb_classes=self.out_dim)
		super().validation_step((x, y), batch_idx=batch_idx)

	def test_step(self, batch, batch_idx):
		x, y = batch
		y = seg2onehot(y, nb_classes=self.out_dim)
		super().test_step((x, y), batch_idx=batch_idx)


class PL_IODA_CNN_MLP(PL_IODA_Base):
	def instantiate_network(self, params: dict) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
		"""
			(x_to_x, y_to_y, link_x_to_y, link_y_to_x)
		"""
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_cnn_mlp_native(params)
			return x_to_x, y_to_y, link_x_to_y, link_y_to_x
		else:
			return get_cnn_mlp(params)
