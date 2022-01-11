import torch
import wandb
from pytorch_lightning import Callback
from torch.utils.data import Dataset
from skimage.draw import line

from Utils.utils import _get_samples
from src.Model_Pix2Pix_CNN_MLP import PL_Pix2Pix_CNN_MLP
from src.Model_Pix2Pix_General import Pix2Pix_Output

line_code = 1
gt_code = 2
pred_code = 3

def create_draw(y: torch.Tensor, pred_y: torch.Tensor, batch_size: int) -> torch.Tensor:
	y_img = torch.zeros(size=[batch_size, 64, 64]).reshape([batch_size, 64, 64])
	for elem in range(len(y)):
		points_y = y[elem]
		points_pred = pred_y[elem]
		for point_i in range(68):
			coord_gt_y, coord_gt_x = points_y[point_i]
			coord_pred_y, coord_pred_x = points_pred[point_i]
			rr, cc = line(coord_gt_x, coord_gt_y, coord_pred_x, coord_pred_y)
			y_img[elem, rr, cc] = line_code

	for elem in range(len(y)):
		points_y = y[elem]
		points_pred = pred_y[elem]
		for point_i in range(68):
			coord_gt_y, coord_gt_x = points_y[point_i]
			coord_pred_y, coord_pred_x = points_pred[point_i]
			y_img[elem, coord_gt_x, coord_gt_y] = gt_code
			y_img[elem, coord_pred_x, coord_pred_y] = pred_code

	return y_img



class SampleLogger_Pix2Pix(Callback):
	def __init__(
		self,
		train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset,
		train_samples_cpt: int, valid_samples_cpt: int, test_samples_cpt: int,
		epoch_frequency: int,
	):
		super(SampleLogger_Pix2Pix, self).__init__()

		def sample2tensor(samples):
			a, b = samples
			return torch.stack(a), torch.stack(b)

		self.train_samples = sample2tensor(_get_samples(train_dataset, train_samples_cpt))
		self.valid_samples = sample2tensor(_get_samples(valid_dataset, valid_samples_cpt))
		self.test_samples = sample2tensor(_get_samples(test_dataset, test_samples_cpt))

		self.epoch_frequency = epoch_frequency

		self.cpt_epoch = -1

	def _log_samples(self, samples: torch.Tensor, model: PL_Pix2Pix_CNN_MLP, prefix: str):
		x, y = samples
		x = x.to(model.device)
		y = y.to(model.device)
		batch_size, _, _, _ = x.shape

		with torch.no_grad():
			pred: Pix2Pix_Output = model((x, y))

		# but back x in the usual image format
		x = pred.x.permute([0, 2, 3, 1])
		# put back the annotation in there real space relatively to the input image
		# the bigger index possible is actually 63
		y = (pred.y * 63).int().reshape([-1, 68, 2])
		pred_y = (pred.fake_y * 63).int().reshape([-1, 68, 2])

		# create an image with a line draw btw the GT and the pred
		y_img = create_draw(y, pred_y, batch_size)

		images = x
		masks  = y_img
		mask_caption = ''

		class_labels = {
			0: 'None',
			line_code: 'Error Line',
			gt_code: 'GT',
			pred_code: 'Prediction',
		}
		for i in range(batch_size):
			img = images[i].cpu().numpy()
			mask = masks[i].cpu().numpy()
			wandb.log({
				f'{prefix}/images': wandb.Image(img, caption='(x, xy)',
				                                masks={mask_caption: {'mask_data': mask, 'class_labels': class_labels}}),
			})

	def on_train_epoch_end(self, trainer, pl_module: PL_Pix2Pix_CNN_MLP) -> None:
		self.cpt_epoch += 1
		if self.cpt_epoch % self.epoch_frequency != 0:
			return
		self._log_samples(self.train_samples, pl_module, 'train')

	def on_validation_epoch_end(self, trainer, pl_module: PL_Pix2Pix_CNN_MLP) -> None:
		if self.cpt_epoch % self.epoch_frequency != 0:
			return
		self._log_samples(self.valid_samples, pl_module, 'valid')

	def on_test_epoch_end(self, trainer, pl_module: PL_Pix2Pix_CNN_MLP) -> None:
		self._log_samples(self.test_samples, pl_module, 'test')
