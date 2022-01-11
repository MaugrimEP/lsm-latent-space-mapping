import time
import random
import numpy as np
import torch
import wandb
from torch.nn import functional as F
from torch.utils.data import Dataset
from dataclasses import field


def return_factory(param):
	return field(default_factory=lambda: param)


def seg2onehot(y: torch.Tensor, nb_classes) -> torch.Tensor:
	y = F.one_hot(y.to(torch.int64), nb_classes).transpose(1, 4).squeeze(-1).type(torch.float32)
	return y


def timestamp() -> str:
	return time.strftime("%Y-%m-%d_%H-%M-%S")


def mode2mask(mode: int) -> np.ndarray:
	"""
	Return the mask which we will multiply the loss with, will be used in the dataset
	"""
	mask_list = [
		None,
		[1, 1, 1],  # XY
		[1, 0, 0],  # X
		[0, 1, 0],  # Y
	]
	return np.array(mask_list[mode])


def mask2mode(mode: np.ndarray) -> int:
	mask_list = {
		(1, 1, 1): 1,  # XY
		(1, 0, 0): 2,  # X
		(0, 1, 0): 3,  # Y
	}
	return mask_list[tuple(mode.tolist())]


def _get_samples(dataset: Dataset, sample_cpt: int, fail_if_not_enough: bool = False):
	available_sample_count = len(dataset)
	if available_sample_count < sample_cpt:
		if fail_if_not_enough:
			raise Exception(f"Not enough sample exception, attempt to sample :{sample_cpt} when only {available_sample_count=} available")
		sample_cpt = available_sample_count

	if sample_cpt == -1:
		indexes = range(len(dataset))
	else:
		indexes = random.sample(range(len(dataset)), sample_cpt)

	samples = [dataset[i] for i in indexes]
	if len(samples[0])==3:
		x, y, _ = zip(*samples)
	else:
		x, y = zip(*samples)
	batched_x = x
	batched_y = y

	return batched_x, batched_y


# region utils for main
def get_wandb_init(params: dict):
	run = wandb.init(
		project=params['project_name'],
		name=params['wdb_name'],
		config=params,
		mode='offline' if params['offline'] else 'online',
		tags=params['tags'],
	)
	return run

# endregion
