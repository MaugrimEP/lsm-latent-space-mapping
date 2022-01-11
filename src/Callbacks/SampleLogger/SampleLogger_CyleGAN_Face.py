from typing import Tuple
import torch
import wandb
from pytorch_lightning import Callback
from torch.utils.data import Dataset
import numpy as np
from torch.nn import functional as F
from torchvision.utils import make_grid

from Utils.utils import _get_samples
from src.Callbacks.SampleLogger.SampleLogger_IODA_FACE import create_draw
from src.Model_CycleGAN_General import CycleGAN_Output, PL_CycleGAN_Base


class_labels = {
	0: 'None',
	1: 'Error Line',
	2: 'GT',
	3: 'Prediction',
}


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
			pred: CycleGAN_Output = model((x, y))

		# (x, y->xy) ; (yx, y->y)
		# (xx, y->y) ; (x ,y->yy)
		f_put_back = lambda y: (y * 63).int().reshape([-1, 68, 2])

		yx = pred.yx
		xx = pred.xx

		y = f_put_back(y)
		yy = f_put_back(pred.yy)
		xy = f_put_back(pred.xy)

		y_2_xy = create_draw(y, xy, len(y))
		y_2_y  = create_draw(y, y , len(y))
		y_2_yy = create_draw(y, yy, len(y))

		imgs_i = [make_grid(
			[x_i, yx_i,
			 xx_i, x_i], nrow=2
		).permute(1, 2, 0) for x_i, yx_i, xx_i in zip(x, yx, xx)]
		masks_i = [make_grid(
			[y_2_xy_i, y_2_y_i,
			 y_2_y_i, y_2_yy_i], nrow=2
		)[0]  # sample first dim, bcs this function is ...
		           for y_2_xy_i, y_2_y_i, y_2_yy_i in zip(y_2_xy, y_2_y, y_2_yy)]
		wandb.log({
			f'{prefix}/epoch': model.current_epoch,
			f'{prefix}/images': [wandb.Image(
				imgs_i[i].cpu().numpy(),
				caption='(x, y->xy) ; (yx, y->y) ; (xx, y->y) ; (x ,y->yy)',
				masks={'mask': {'mask_data': masks_i[i].cpu().numpy(), 'class_labels': class_labels}}) for i in range(len(imgs_i))],
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
