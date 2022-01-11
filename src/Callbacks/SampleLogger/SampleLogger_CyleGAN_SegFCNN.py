from typing import Tuple
import torch
import wandb
from pytorch_lightning import Callback
from torch.utils.data import Dataset
import numpy as np
from torch.nn import functional as F
from torchvision.utils import make_grid

from Utils.utils import _get_samples
from src.Model_CycleGAN_General import CycleGAN_Output, PL_CycleGAN_Base


class SampleLogger_CycleGAN(Callback):
	def __init__(
			self,
			train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset,
			train_samples_cpt: int, valid_samples_cpt: int, test_samples_cpt: int,
			epoch_frequency: int,
	):
		super(SampleLogger_CycleGAN, self).__init__()

		def sample2tensor(samples):
			a, b = samples
			return torch.stack(a), torch.stack(b)

		self.train_samples = sample2tensor(_get_samples(train_dataset, train_samples_cpt))
		self.valid_samples = sample2tensor(_get_samples(valid_dataset, valid_samples_cpt))
		self.test_samples  = sample2tensor(_get_samples(test_dataset , test_samples_cpt))

		self.epoch_frequency = epoch_frequency
		self.cpt_epoch = -1

	def _log_samples(self, samples: Tuple[np.ndarray, np.ndarray], model: PL_CycleGAN_Base, prefix: str):
		x, y = samples
		x = x.to(model.device)
		y = y.to(model.device)
		batch_size, _, _, _ = x.shape

		with torch.no_grad():
			pred: CycleGAN_Output = model((x, F.one_hot(y.to(torch.int64), model.out_dim).transpose(1, 4).squeeze(-1).type(torch.float32)))

		# bs, c, h, w
		# x(y) x(xy) yx(y)
		# xx(y) x(yy)
		x = pred.x
		xx = pred.xx
		yx = pred.yx

		y = pred.y
		yy = pred.yy
		xy = pred.xy

		imgs_i = [make_grid(
			[x_i, x_i, yx_i,
			 xx_i, x_i], nrow=3
		).permute(1, 2, 0) for x_i, yx_i, xx_i in zip(x, yx, xx)]
		masks_i = [make_grid(
			[y_i, xy_i, y_i,
			 y_i, yy_i], nrow=3
		) for y_i, xy_i, yy_i in zip(y, xy, yy)]
		masks_i = [torch.argmax(masks_i[i], dim=0) for i in range(len(masks_i))]

		imgs_i  = [imgs_i[i].cpu().numpy() for i in range(len(imgs_i))]
		masks_i = [masks_i[i].cpu().numpy() for i in range(len(masks_i))]

		wandb.log({
			f'{prefix}/epoch': model.current_epoch,
			f'{prefix}/images': [wandb.Image(imgs_i[i],
											caption=f'{i}: (x, y) ; (x, xy) ; (yx, y) ; (xx ,y) ; (x ,yy)',
											masks={'mask': {'mask_data': masks_i[i]}}) for i in range(len(imgs_i))],
	})

	def on_train_epoch_end(self, trainer, pl_module: PL_CycleGAN_Base) -> None:
		self.cpt_epoch += 1
		if self.cpt_epoch % self.epoch_frequency != 0:
			return
		with torch.no_grad():
			self._log_samples(self.train_samples, pl_module, 'train')

	def on_validation_epoch_end(self, trainer, pl_module: PL_CycleGAN_Base) -> None:
		if self.cpt_epoch % self.epoch_frequency != 0:
			return
		self._log_samples(self.valid_samples, pl_module, 'valid')

	def on_test_epoch_end(self, trainer, pl_module: PL_CycleGAN_Base) -> None:
		self._log_samples(self.test_samples, pl_module, 'test')
