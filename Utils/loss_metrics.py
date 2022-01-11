from typing import Callable

import torch
import torch.nn.functional as F

from datasets.Face.Face_dataset_Utils import inter_ocular_dist


def iou_torch(outputs: torch.Tensor, labels: torch.Tensor, remove_background: bool = True, reduction: bool = True):
	"""
	Not a class wise IoU
	:param outputs: BATCH x CLASSES x H x W: binarized: LongTensor
	:param labels: BATCH x CLASSES x H x W: LongTensor
	:param remove_background: if we compute the iou on the background or not
	:param reduction: if compute batchsize wise or note
	:return:
	"""
	if remove_background:
		nb_classes = labels.shape[1]
		to_select  = [i for i in range(1, nb_classes)]
		outputs = outputs[:, to_select]
		labels  = labels[:, to_select]

	intersection = (outputs & labels).float().sum([2, 3])
	union = (outputs | labels).float().sum([2, 3])

	iou = (intersection) / (union)

	if reduction:
		return iou.mean()
	else:
		return iou


def NRMSE(outputs: torch.Tensor, labels: torch.Tensor, reduction: bool = True):
	"""
	:param outputs: BATCH x 68*2
	:param labels: BATCH x 68*2
	:return:
	"""
	outputs = outputs.detach().reshape([-1, 68, 2])
	labels = labels.detach().reshape([-1, 68, 2])

	# batch size x 68 x 2
	point_wise_mse = F.mse_loss(outputs, labels, reduction='none')  # batch size x 68 x 2
	point_wise_mse = torch.sum(point_wise_mse, dim=-1)  # batch size x 68
	point_wise_rmse = torch.sqrt(point_wise_mse)  # batch size x 68
	# batch size x 1
	element_wise_rmse = torch.mean(point_wise_rmse, dim=1)  # batch size x 1

	element_wise_inter_ocular_distance = inter_ocular_dist(labels)
	element_wise_normalized = element_wise_rmse/element_wise_inter_ocular_distance

	if reduction:
		NRMSE_val = element_wise_normalized.mean()
	else:
		NRMSE_val = element_wise_normalized.reshape(-1, 1)

	return NRMSE_val


def get_loss(loss_name: str) -> Callable:
	return {
		'mse': mse_loss_c,
		'pairwise': pairwise_loss,
		'ce': cross_entropy_2d,
		'ce_no_w': cross_entropy_2d_not_weighted,
		'dice': dice_loss_2d,
		'dice_no_w': dice_loss_2d_not_weighted,
		'l1': l1_loss_c,
		'emd': earth_mover_distance,
		'gan_loss': gan_loss,
		'confusion_loss': confusion_loss,
		'nrmse': nrmse,
		'miou': miou,
		'BCE': A_BCE,
		'BCE_NS': A_BCE_NS,
		'SOFTPLUS': A_Softplus,
		'MSE': A_MSE,
	}[loss_name]


class nrmse:
	def __init__(self, reduction: bool = True):
		self.reduction = reduction
		self.name = 'nrmse'
		self.best = 'min'

	def __call__(self, hat_y, y):
		return NRMSE(hat_y, y, reduction=self.reduction)


class miou:
	def __init__(self, reduction: bool = True):
		self.reduction = reduction
		self.name = 'miou'
		self.best = 'max'

	def __call__(self, hat_y, y):
		return iou_torch((hat_y > 0.5).int(), y.int(), reduction=self.reduction)


class earth_mover_distance:
	def __init__(self, reduction: bool = True):
		self.name = 'earth_mover_distance'
		self.reduction = reduction

	def __call__(self, pred: torch.Tensor, real: torch.Tensor):
		batch_size = pred.shape[0]
		pred = pred.reshape([batch_size, -1])
		real = real.reshape([batch_size, -1])
		if self.reduction:
			return pred.mean(), - real.mean()
		else:
			return pred, -real


class gan_loss:
	def __init__(self, reduction: bool = True):
		self.name = 'gan_loss'
		self.reduction = reduction

	def __call__(self, fake: torch.Tensor, real: torch.Tensor):
		batch_size = fake.shape[0]
		fake = fake.reshape([batch_size, -1])
		real = real.reshape([batch_size, -1])

		fake_loss = F.binary_cross_entropy(fake, torch.zeros_like(fake), reduction='none')
		real_loss = F.binary_cross_entropy(real, torch.ones_like(real), reduction='none')
		if self.reduction:
			return fake_loss.mean(), real_loss.mean()
		else:
			return fake_loss, real_loss


class confusion_loss:
	def __init__(self, reduction: bool = True):
		self.name = 'confusion_loss'
		self.reduction = reduction

	def __call__(self, fake: torch.Tensor, real: torch.Tensor):
		batch_size = fake.shape[0]
		fake = fake.reshape([batch_size, -1])
		real = real.reshape([batch_size, -1])

		# minus torch.log(torch.Tensor([0.5]))
		fake_loss = F.binary_cross_entropy(fake, 0.5 * torch.ones_like(fake), reduction='none') - 0.69
		real_loss = F.binary_cross_entropy(real, 0.5 * torch.ones_like(real), reduction='none') - 0.69
		if self.reduction:
			return fake_loss.mean(), real_loss.mean()
		else:
			return fake_loss, real_loss


class mse_loss_c:
	def __init__(self, reduction: bool = True):
		self.reduction = reduction
		self.name = 'mse'
		self.best = 'min'

	def __call__(self, pred: torch.Tensor, y: torch.Tensor):
		return F.mse_loss(
			pred.flatten(start_dim=1),
			y.flatten(start_dim=1),
			reduction='mean' if self.reduction else 'none'
		)

class pairwise_loss:
	def __init__(self, reduction: bool = True):
		self.reduction = reduction
		self.name = 'pairwise_loss'
		self.best = 'min'

	def __call__(self, pred: torch.Tensor, y: torch.Tensor):
		loss = F.pairwise_distance(
			pred.flatten(start_dim=1),
			y.flatten(start_dim=1),
			p=2,
		)
		if self.reduction:
			loss = loss.mean()
		return loss

class l1_loss_c:
	def __init__(self, reduction: bool = True):
		self.reduction = reduction
		self.name      = 'l1_loss_c'
		self.best = 'min'

	def __call__(self, pred: torch.Tensor, y: torch.Tensor):
		loss = F.l1_loss(
			pred.flatten(start_dim=1),
			y.flatten(start_dim=1),
			reduction='mean' if self.reduction else 'none',
		)
		return loss

class cross_entropy_2d:
	def __init__(self, reduction: bool = True, weighted: bool = True):
		self.reduction = reduction
		self.name = 'cross_entropy_2d'
		self.weighted = weighted
		self.best = 'min'

	def __call__(self, pred: torch.Tensor, y: torch.Tensor):
		if self.weighted:
			img_size = y.shape[2] * y.shape[3]
			weights  = img_size / torch.sum(y, dim=[0, 2, 3])
			loss = F.cross_entropy(pred, y, reduction='mean' if self.reduction else 'none', weight=weights)
		else:
			loss = F.cross_entropy(pred, y, reduction='mean' if self.reduction else 'none')
		return loss

class cross_entropy_2d_not_weighted:
	def __init__(self, reduction: bool = True, weighted: bool = True):
		self.proxy = cross_entropy_2d(reduction=reduction, weighted=False)
		self.reduction = reduction
		self.name = 'cross_entropy_2d_not_weighted'
		self.weighted = False
		self.best = 'min'

	def __call__(self, pred: torch.Tensor, y: torch.Tensor):
		return self.proxy(pred=pred, y=y)

class dice_loss_2d:
	def __init__(self, reduction: bool = True, weighted: bool = True, eps: float = 1e-7):
		self.reduction = reduction
		self.name = 'dice_loss_2d'
		self.weighted = weighted
		self.eps = eps
		self.best = 'min'

	def __call__(self, pred: torch.Tensor, y: torch.Tensor):
		"""
		:param pred: Logits
		:param y: one-hot
		:param weighted:
		:param reduction:
		:param eps:
		:return: Return ([batch_size, nbclass], nbclass) is reduce is False, or [1] otherwise
		"""
		# num_classes = pred.shape[1]
		# true_1_hot = torch.eye(num_classes)[y.squeeze(1)].permute(0, 3, 1, 2).float()
		true_1_hot = y

		probas = F.softmax(pred, dim=1)

		true_1_hot = true_1_hot.type(pred.type())
		dims = tuple(range(2, y.ndimension()))

		intersection = torch.sum(probas * true_1_hot, dims)
		cardinality = torch.sum(probas + true_1_hot, dims)

		dice_loss = 1 - (2. * intersection / (cardinality + self.eps))

		if self.weighted:
			img_size  = y.shape[2] * y.shape[3]
			weights   = img_size / torch.sum(true_1_hot, dim=[0, 2, 3])
			dice_loss = dice_loss * weights

		if self.reduction:
			dice_loss = dice_loss.mean()

		return dice_loss


class dice_loss_2d_not_weighted:
	def __init__(self, reduction: bool = True, weighted: bool = True):
		self.proxy = dice_loss_2d(reduction=reduction, weighted=False)
		self.reduction = reduction
		self.name = 'dice_loss_2d_not_weighted'
		self.weighted = False
		self.best = 'min'

	def __call__(self, pred: torch.Tensor, y: torch.Tensor):
		return self.proxy(pred=pred, y=y)


class Adversarial_loss:
	def __init__(self, is_confusing: bool = False):
		self.is_confusing = is_confusing
		self.target_reals = 1. if not is_confusing else 0.5
		self.target_fakes = 0. if not is_confusing else 0.5

	def g_fake(self, fakes_decision):
		raise NotImplemented

	def d_fakes(self, fakes_decision):
		raise NotImplemented

	def d_reals(self, reals_decision):
		raise NotImplemented

	def get_triplet(self, fakes_decision, reals_decision):
		g_f = self.g_fake(fakes_decision)
		d_f = self.d_fakes(fakes_decision)
		d_r = self.d_reals(reals_decision)

		return g_f, (d_f, d_r)


class A_BCE(Adversarial_loss):
	def g_fake(self, fakes_decision):
		return - F.binary_cross_entropy_with_logits(
			fakes_decision,
			torch.full_like(fakes_decision, self.target_fakes),
			reduction='none',
		)

	def d_fakes(self, fakes_decision):
		return F.binary_cross_entropy_with_logits(
			fakes_decision,
			torch.full_like(fakes_decision, self.target_fakes),
			reduction='none',
		)

	def d_reals(self, reals_decision):
		return F.binary_cross_entropy_with_logits(
			reals_decision,
			torch.full_like(reals_decision, self.target_reals),
			reduction='none',
		)


class A_BCE_NS(Adversarial_loss):
	def g_fake(self, fakes_decision):
		return F.binary_cross_entropy_with_logits(
			fakes_decision,
			torch.full_like(fakes_decision, self.target_reals),
			reduction='none',
		)

	def d_fakes(self, fakes_decision):
		return F.binary_cross_entropy_with_logits(
			fakes_decision,
			torch.full_like(fakes_decision, self.target_fakes),
			reduction='none',
		)

	def d_reals(self, reals_decision):
		return F.binary_cross_entropy_with_logits(
			reals_decision,
			torch.full_like(reals_decision, self.target_reals),
			reduction='none',
		)


class A_Softplus(Adversarial_loss):
	def g_fake(self, fakes_decision):
		if self.is_confusing:
			raise NotImplemented
		return F.softplus(-fakes_decision)

	def d_fakes(self, fakes_decision):
		if self.is_confusing:
			raise NotImplemented
		return F.softplus(fakes_decision)

	def d_reals(self, reals_decision):
		if self.is_confusing:
			raise NotImplemented
		return F.softplus(-reals_decision)


class A_MSE(Adversarial_loss):
	def g_fake(self, fakes_decision):
		return F.mse_loss(
			fakes_decision,
			torch.full_like(fakes_decision, self.target_reals),
			reduction='none',
		)

	def d_fakes(self, fakes_decision):
		return F.mse_loss(
			fakes_decision,
			torch.full_like(fakes_decision, self.target_fakes),
			reduction='none',
		)

	def d_reals(self, reals_decision):
		return F.mse_loss(
			reals_decision,
			torch.full_like(reals_decision, self.target_reals),
			reduction='none',
		)


def _get_adversarial_loss(loss_name: str) -> Adversarial_loss:
	return {
		'BCE': A_BCE(),
		'BCE_NS': A_BCE_NS(),
		'SOFTPLUS': A_Softplus(),
		'MSE': A_MSE(),
	}[loss_name]
