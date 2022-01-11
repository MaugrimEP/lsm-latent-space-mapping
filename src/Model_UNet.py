from __future__ import annotations
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from Utils.loss_metrics import iou_torch
from Utils.utils import seg2onehot
from src.Networks.UNet import UNet


class UNet_pl(pl.LightningModule):
	def __init__(
			self,
			input_dimension: int,
			output_dimension: int,
			out_sz: List[int],
			learning_rate: float,
	):
		super().__init__()

		self.in_dim = input_dimension
		self.out_dim = output_dimension
		self.lr = learning_rate

		self.model = UNet(
			in_dim=input_dimension,
			out_class=output_dimension,
			retain_dim=True,
			out_sz=out_sz,
		)


	def forward(self, x):
		return self.model(x)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr)

	def compute_loss(self, y, pred):
		l_xy = F.cross_entropy(pred, y)
		return l_xy

	def _log(self, prefix: str, miou_xy: torch.Tensor, loss: torch.Tensor):
		self.log(f'{prefix}/iou_xy', miou_xy, on_step=False, on_epoch=True, prog_bar=True)
		self.log(f'{prefix}/loss_translated_x_y' , loss , on_step=False, on_epoch=True, prog_bar=True)
		self.log(f'{prefix}/sum_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

	def training_step(self, train_batch, batch_idx, optimizer_idx=0):
		prefix = "train"

		x, y, m = train_batch
		y = seg2onehot(y, self.out_dim)
		pred = self(x)

		loss = self.compute_loss(y, pred)
		miou_xy = iou_torch((pred > 0.5).int(), y.int())

		self._log(prefix=prefix, miou_xy=miou_xy, loss=loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		prefix = "valid"

		x, y, m = val_batch
		y = seg2onehot(y, self.out_dim)
		pred = self(x)

		loss = self.compute_loss(y, pred)
		miou_xy = iou_torch((pred > 0.5).int(), y.int())

		self._log(prefix=prefix, miou_xy=miou_xy, loss=loss)

	def test_step(self, test_batch, batch_idx):
		prefix = "test"

		x, y, m = test_batch
		y = seg2onehot(y, self.out_dim)
		pred = self(x)

		loss = self.compute_loss(y, pred)
		miou_xy = iou_torch((pred > 0.5).int(), y.int())

		self._log(prefix=prefix, miou_xy=miou_xy, loss=loss)
