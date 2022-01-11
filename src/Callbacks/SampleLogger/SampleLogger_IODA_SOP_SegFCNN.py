from typing import Tuple
import torch
import wandb
from pytorch_lightning import Callback
from torch.utils.data import Dataset

from Utils.utils import seg2onehot, _get_samples
from src.Model_IODA_General import PL_IODA_FULL_CNN, IODA_Output


class SampleLogger_Sarcopedia_IODA(Callback):
	def __init__(
		self,
		train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset,
		train_samples_cpt: int, valid_samples_cpt: int, test_samples_cpt: int,
		epoch_frequency: int,
	):
		super(SampleLogger_Sarcopedia_IODA, self).__init__()

		def sample2tensor(samples):
			a, b = samples
			return torch.stack(a), torch.stack(b)

		self.train_samples = sample2tensor(_get_samples(train_dataset, train_samples_cpt))
		self.valid_samples = sample2tensor(_get_samples(valid_dataset, valid_samples_cpt))
		self.test_samples = sample2tensor(_get_samples(test_dataset, test_samples_cpt))

		self.epoch_frequency = epoch_frequency

		self.cpt_epoch = -1

	def _log_samples(self, samples: Tuple[torch.Tensor, torch.Tensor], model, prefix: str):
		x, y = samples
		x = x.to(model.device)
		y = y.to(model.device)
		batch_size, c, h, w = x.shape

		with torch.no_grad():
			pred = model((x, seg2onehot(y, nb_classes=model.out_dim)))
		y = y.reshape([batch_size, h, w])

		mask_y  = y
		mask_yy = torch.argmax(pred.yy, dim=1)
		mask_xy = torch.argmax(pred.xy, dim=1)

		caption = 'x(y) xx(y) x(yy) x(xy)'
		images = torch.cat([pred.x, pred.xx, pred.x , pred.x ], dim=3)
		masks  = torch.cat([mask_y, mask_y , mask_yy, mask_xy], dim=2)

		images = images.cpu()
		masks  = masks.cpu()

		if hasattr(model, 'mode'):
			key = f'{prefix}/{model.mode}/images'
		else:
			key = f'{prefix}/images'
		wandb.log({
			key: [wandb.Image(img, caption=caption, masks={'mask': {'mask_data': mask.numpy()}}) for img, mask in zip(images, masks)],
		})

	def on_train_epoch_end(self, trainer, pl_module) -> None:
		self.cpt_epoch += 1
		if self.cpt_epoch % self.epoch_frequency != 0:
			return
		self._log_samples(self.train_samples, pl_module, 'train')

	def on_validation_epoch_end(self, trainer, pl_module) -> None:
		if self.cpt_epoch % self.epoch_frequency != 0:
			return
		self._log_samples(self.valid_samples, pl_module, 'valid')

	def on_test_epoch_end(self, trainer, pl_module) -> None:
		self._log_samples(self.test_samples, pl_module, 'test')
